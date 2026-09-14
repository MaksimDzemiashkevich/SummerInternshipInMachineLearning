import io
import re
import json
import logging
from rapidfuzz import fuzz
from functools import lru_cache
from typing import Literal, Annotated
from operator import add

import pandas as pd
from dotenv import load_dotenv

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Depends, UploadFile, File
from fastapi.responses import Response

from langgraph.graph import START, StateGraph, MessagesState, END
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import RemoveMessage, ToolMessage

from .modules.config import AppConfig, Settings, build_app_config, hybrid_search
from .modules.system_prompts import SYSTEM_PROMPT1
from .modules.tools import array_of_tools

TOOLS_BY_NAME = {t.name: t for t in array_of_tools}

load_dotenv()

logger = logging.getLogger("normalization_pipeline")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(handler)

RULES_PER_CHUNK = 12
RETRIEVAL_K = 4


class GraphState(MessagesState):
    index_row_table: int
    attribute: str
    value: str
    original_value: str
    deterministic_result: str | None
    new_rows: Annotated[list[dict], add]
    data_frame: pd.DataFrame
    remaining_chunks: list[str]


def split_into_rule_chunks(text: str, rules_per_chunk: int = RULES_PER_CHUNK) -> list[str]:

    if not text:
        return []

    # Делим по ПРАВИЛАМ, а не по предложениям: каждое правило — отдельная строка
    # (в body/global_rules правила склеены через '\n'), поэтому режем по переносам строк.
    units = [u.strip() for u in text.split("\n") if u.strip()]

    if not units:
        return []

    chunks = []
    for i in range(0, len(units), rules_per_chunk):
        chunk_text = "\n".join(units[i:i + rules_per_chunk])
        chunks.append(chunk_text)

    return chunks


def _is_ambiguous_answer(answer) -> bool:
    """Whether the model flagged the row as ambiguous (the AMBIGUOUS token)."""
    text = str(answer).strip()
    # Strip markdown code fences, quotes and trailing punctuation models often add.
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    text = text.strip('`"\'*_#,.()[]:; ')
    return text.upper() == "AMBIGUOUS"


def _parse_result(answer) -> str:
    """Извлекает итоговое значение из структурированного ответа модели (fallback на сырой текст)."""
    text = str(answer).strip()
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "result" in obj:
            return str(obj["result"])
    except Exception:
        pass
    return text


def _normalize_phrase(text: str) -> str:
    """Нормализует фразу для сравнения: нижний регистр, пунктуация→пробел, порядок слов не важен."""
    text = text.lower()
    text = re.sub(r"[()\-,]", " ", text)
    return " ".join(sorted(text.split()))


def _norm_attr(text: str) -> str:
    """Нормализует имя атрибута для сравнения (без сортировки слов, только регистр/пробелы)."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _attribute_is_excluded(attribute: str, excluded: list[str]) -> bool:
    """Проверяет, попадает ли атрибут строки в список исключённых (по подстроке)."""
    a = _norm_attr(attribute)
    if not a or not excluded:
        return False
    for e in excluded:
        e_norm = _norm_attr(e)
        if not e_norm:
            continue
        if a in e_norm or e_norm in a:
            return True
    return False


def _attribute_matches(attribute: str, owner: str) -> bool:
    """Совпадает ли атрибут строки с атрибутом-владельцем keyword (по подстроке)."""
    a = _norm_attr(attribute)
    o = _norm_attr(owner)
    if not a or not o:
        return True  # нет данных о владельце — применяем (backward compat)
    return o in a or a in o


def _match_render_template(attribute: str, render_templates: dict) -> dict | None:
    """Находит шаблон рендера для атрибута (по имени группы + алиасам, подстрока)."""
    a = _norm_attr(attribute)
    if not a:
        return None
    for attr, tmpl in render_templates.items():
        for c in [attr] + list(tmpl.get("aliases", [])):
            c_norm = _norm_attr(c)
            if c_norm and (c_norm in a or a in c_norm):
                return tmpl
    return None


def _apply_render_template(answer: str, template: dict) -> str:
    """Детерминированно переформатирует ответ под шаблон (decimals + suffix)."""
    numbers = re.findall(r"-?\d+(?:\.\d+)?", answer)
    if len(numbers) != 1:
        return answer  # диапазон/сложное значение — оставляем как есть (fallback)
    num = float(numbers[0])
    return f"{num:.{template['decimals']}f}" + template["suffix"]


FUZZY_THRESHOLD = 90


def match_closed_register(value: str, table: dict[str, str], threshold: int = FUZZY_THRESHOLD) -> str | None:
    """Fuzzy-match значения против 'фраза таблицы -> код' одного закрытого регистра."""
    if not table:
        return None
    norm_value = _normalize_phrase(value)
    best_key, best_score = None, 0
    for key in table:
        score = fuzz.token_sort_ratio(norm_value, _normalize_phrase(key))
        if score > best_score:
            best_key, best_score = key, score
    if best_score >= threshold:
        return table[best_key]
    prefix = next(iter(table.values()))[:3].rstrip("-0123456789")
    return f"{prefix} pending registration: {value.strip()}"


def apply_deterministic(value: str, rule_set, attribute: str) -> str | None:
    """Детерминированные преобразования. Возвращает финальное значение, если оно полностью
    закрыто детерминированными/фразовыми правилами; иначе None (идём в LLM)."""
    result = value.strip()

    # 0) атрибут исключён из реформатирования — возвращаем без изменений, LLM не вызываем.
    if _attribute_is_excluded(attribute, rule_set.excluded_attributes):
        return result

    # 1) точное фразовое совпадение (phrase_rules + бесчисловые deterministic).
    phrase_keys = {}
    for k, v in rule_set.phrase_map.items():
        phrase_keys[_normalize_phrase(k)] = v
    for k, v in rule_set.deterministic.items():
        if not re.search(r"\d", k):  # «чистые» фразы без числа
            phrase_keys[_normalize_phrase(k)] = v

    if phrase_keys:
        parts = result.split(";")
        mapped = []
        ok = True
        for part in parts:
            p = part.strip()
            if not p:
                mapped.append("")
                continue
            code = phrase_keys.get(_normalize_phrase(p))
            if code is None:
                ok = False
                break
            mapped.append(code)
        if ok:
            return "; ".join(mapped)

    # 2) fuzzy-match против закрытого регистра, соответствующего атрибуту.
    prefix = rule_set.attribute_to_register.get(attribute.strip().lower())
    if prefix:
        matched = match_closed_register(result, rule_set.closed_registers.get(prefix, {}))
        if matched is not None:
            return matched

    # 3) keyword → root: всё значение целиком должно быть «keyword число» / «число keyword».
    for keyword in sorted(rule_set.deterministic, key=len, reverse=True):
        # attribute-aware: применяем keyword только к его атрибуту-владельцу
        if not _attribute_matches(attribute, rule_set.deterministic_attributes.get(keyword, "")):
            continue
        root = rule_set.deterministic[keyword]
        token = rf"(?:{re.escape(keyword)}|{re.escape(root)})"
        num = r"(\d+[A-Za-z]*)"
        m = (re.fullmatch(rf"\s*{token}\s*{num}\s*", result, re.IGNORECASE)
             or re.fullmatch(rf"\s*{num}\s*{token}\s*", result, re.IGNORECASE))
        if m:
            num_part = m.group(1)
            nf = rule_set.deterministic_order.get(keyword, rule_set.number_first)
            return (num_part + root) if nf else (root + num_part)

    return None


def start_row(state: GraphState, config: RunnableConfig) -> GraphState:

    update: dict = {}
    data_frame = state["data_frame"]
    rule_set = config["configurable"]["rule_set"]

    # --- фиксируем результат ПРЕДЫДУЩЕЙ строки ---
    if state["index_row_table"] != 0:
        if state.get("deterministic_result") is not None:
            last_answer = state["deterministic_result"]
            is_ambiguous = False
        elif state.get("messages"):
            last_answer = _parse_result(state["messages"][-1].content)
            is_ambiguous = _is_ambiguous_answer(last_answer)
        else:
            last_answer = state["original_value"]
            is_ambiguous = False
        new_row = {
            "Attribute": state["attribute"],
            "Value": state["original_value"],
            "Rules_Value": state["original_value"] if is_ambiguous else last_answer,
            "Ambiguous": is_ambiguous,
        }
        update["new_rows"] = [new_row]

    # --- настраиваем ТЕКУЩУЮ строку ---
    if state["index_row_table"] < data_frame.shape[0]:
        attribute = data_frame.iloc[state["index_row_table"]]["Attribute"]
        value = data_frame.iloc[state["index_row_table"]]["Value"]

        update["attribute"] = attribute
        update["original_value"] = value
        update["messages"] = [RemoveMessage(id=k.id) for k in state["messages"]]

        det_value = apply_deterministic(value, rule_set, attribute)
        if det_value is not None:
            # полностью детерминировано — LLM не нужен
            update["deterministic_result"] = det_value
            update["value"] = det_value
            update["remaining_chunks"] = []
            logger.info("=" * 80)
            logger.info(
                "Row %d/%d | Attribute: %r | Value: %r | DETERMINISTIC -> %r",
                state["index_row_table"] + 1, data_frame.shape[0], attribute, value, det_value,
            )
        else:
            update["deterministic_result"] = None
            update["value"] = value

            query = attribute
            rule_bodies = hybrid_search(rule_set, query, top_k=RETRIEVAL_K)

            all_chunks: list[str] = []
            if rule_set.global_rules:
                all_chunks.extend(split_into_rule_chunks(rule_set.global_rules))
            for idx, body in enumerate(rule_bodies):
                all_chunks.extend(split_into_rule_chunks(body))
                logger.info("  [SEARCH] Rule %d/%d: %r", idx + 1, len(rule_bodies), body[:120])

            if not all_chunks:
                all_chunks = ["No relevant rules were found in the knowledge base for this attribute."]

            update["remaining_chunks"] = all_chunks
            logger.info("=" * 80)
            logger.info(
                "Row %d/%d | Attribute: %r | Value: %r | chunks: %d",
                state["index_row_table"] + 1, data_frame.shape[0], attribute, value, len(all_chunks)
            )

    return update


def _invoke_with_tools(model, messages):
    """Инвокает модель, прогоняя цикл tool-calls (для юнит-конверсий)."""
    response = model.invoke(messages)
    while getattr(response, "tool_calls", None):
        tool_messages = []
        for tc in response.tool_calls:
            tool_fn = TOOLS_BY_NAME.get(tc["name"])
            if tool_fn is None:
                content = "unknown tool"
            else:
                try:
                    content = str(tool_fn.invoke(tc["args"]))
                except Exception as e:
                    content = f"error: {e}"
            tool_messages.append(ToolMessage(content=content, tool_call_id=tc["id"]))
        messages = messages + [response] + tool_messages
        response = model.invoke(messages)
    return response


def process_chunk(state: GraphState, config: RunnableConfig) -> GraphState:
    model = config["configurable"]["model"]

    chunks = state["remaining_chunks"]
    current_chunk, rest_chunks = chunks[0], chunks[1:]

    is_first_chunk = len(state["messages"]) == 0

    if is_first_chunk:
        user_content = (
            f"Attribute: {state['attribute']}\n"
            f"Value: {state['value']}\n\n"
            f"Relevant rules:\n{current_chunk}"
        )
    else:
        user_content = (
            f"Additional relevant rules for the same attribute/value:\n{current_chunk}\n\n"
            f"If any of these rules apply, update your previously normalized value accordingly. "
            f"Otherwise keep your previous answer unchanged."
        )

    user_msg = {"role": "user", "content": user_content}

    messages = [{"role": "system", "content": SYSTEM_PROMPT1}] + state["messages"] + [user_msg]
    response = _invoke_with_tools(model, messages)

    return {
        "messages": [user_msg, response],
        "remaining_chunks": rest_chunks,
    }


def consolidate(state: GraphState, config: RunnableConfig) -> GraphState:
    """Финальный consolidation-шаг: переоценить ответ по полному набору правил."""
    model = config["configurable"]["model"]
    if not state.get("messages"):
        return {}

    user_msg = {
        "role": "user",
        "content": (
            "You have now seen ALL relevant rules for this attribute/value. "
            "Re-evaluate your current answer against the complete rule set. "
            "If any earlier transformation was applied incorrectly, undo or correct it. "
            "Output only the final standardized value."
        ),
    }
    messages = [{"role": "system", "content": SYSTEM_PROMPT1}] + state["messages"] + [user_msg]
    response = _invoke_with_tools(model, messages)
    return {"messages": [user_msg, response]}


def next_row(state: GraphState) -> GraphState:
    return {"index_row_table": state["index_row_table"] + 1}


def condition_after_start_row(state: GraphState) -> Literal["process_chunk", "next_row", "end_node"]:
    if state["index_row_table"] >= state["data_frame"].shape[0]:
        return "end_node"
    if state.get("deterministic_result") is not None:
        return "next_row"
    return "process_chunk"


def condition_after_process_chunk(state: GraphState) -> Literal["process_chunk", "next_row"]:
    if state["remaining_chunks"]:
        return "process_chunk"
    return "next_row"


def end_node(state: GraphState) -> GraphState:
    return {}


builder = StateGraph(GraphState)

builder.add_node("start_row", start_row)
builder.add_node("process_chunk", process_chunk)
builder.add_node("next_row", next_row)
builder.add_node("end_node", end_node)

builder.add_edge(START, "start_row")
builder.add_conditional_edges("start_row", condition_after_start_row)
builder.add_conditional_edges("process_chunk", condition_after_process_chunk)
builder.add_edge("next_row", "start_row")
builder.add_edge("end_node", END)

graph = builder.compile()


def run_single_rule_set(data_frame: pd.DataFrame, model, rule_set) -> pd.DataFrame:
    initial_state = {
        "messages": [],
        "index_row_table": 0,
        "data_frame": data_frame,
        "remaining_chunks": [],
        "new_rows": [],
        "original_value": "",
        "deterministic_result": None,
    }
    final_state = graph.invoke(
        initial_state,
        config={"configurable": {"model": model, "rule_set": rule_set}},
    )
    return pd.DataFrame(final_state["new_rows"])


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache
def get_app_config() -> AppConfig:
    return build_app_config(Settings())


def _write_excel_with_highlight(df: pd.DataFrame, ambiguous_cols: dict[str, list[bool]]) -> io.BytesIO:
    """Пишет DataFrame в Excel, заливая спорные ячейки жёлтым."""
    from openpyxl import Workbook
    from openpyxl.styles import PatternFill

    wb = Workbook()
    ws = wb.active
    yellow = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")

    ws.append(list(df.columns))
    for _, row in df.iterrows():
        ws.append(row.tolist())

    col_index = {name: i for i, name in enumerate(df.columns, start=1)}
    for col_name, flags in ambiguous_cols.items():
        c = col_index[col_name]
        for r, is_amb in enumerate(flags, start=2):
            if is_amb:
                ws.cell(row=r, column=c).fill = yellow

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf


def check_consistency(result_df: pd.DataFrame, rule_set_names: list[str]) -> None:
    """Логирует warning, если одинаковые по смыслу входы дали разные результаты."""
    for col in rule_set_names:
        key = list(zip(result_df["Attribute"], result_df["Value"].map(_normalize_phrase)))
        seen: dict[tuple, str] = {}
        for k, val in zip(key, result_df[col]):
            if k in seen and seen[k] != val:
                logger.warning(
                    "Inconsistent normalization for %s (col=%s): %r vs %r",
                    k, col, seen[k], val,
                )
            else:
                seen[k] = val


@app.post("/process")
def process_excel(
        file: UploadFile = File(...),
        app_config: AppConfig = Depends(get_app_config),
):
    contents = file.file.read()
    data_frame = pd.read_excel(io.BytesIO(contents))

    result_df = data_frame[["Attribute", "Value"]].copy()
    ambiguous_cols: dict[str, list[bool]] = {}
    for rule_set in app_config.rule_sets:
        col_df = run_single_rule_set(data_frame, app_config.model, rule_set)
        result_df[rule_set.name] = col_df["Rules_Value"].to_list()
        if "Ambiguous" in col_df:
            ambiguous_cols[rule_set.name] = col_df["Ambiguous"].to_list()

    check_consistency(result_df, [rs.name for rs in app_config.rule_sets])

    output_buffer = _write_excel_with_highlight(result_df, ambiguous_cols)

    return Response(
        content=output_buffer.read(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=result.xlsx"},
    )