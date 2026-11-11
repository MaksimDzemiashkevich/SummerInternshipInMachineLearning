"""Граф LangGraph одного документа: строка за строкой, детерминированно или через модель."""
from operator import add
from typing import Literal, Annotated

import pandas as pd
from langchain_core.messages import RemoveMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, StateGraph, MessagesState, END

from .common import _alnum, logger
from .answers import _describe_change, _is_ambiguous_answer, _is_pending_result
from .subjects import _document_covers_row
from .deterministic import RULE_CARDS, apply_deterministic
from .retrieval import _own_rules_text, _unresolved_references, build_rules_text
from .guards import (_align_root_spacing, _apply_render_template, _document_mandates_spacing,


                     _document_word_tokens, _match_render_template, _partial_separator_edit,
                     _rejection_reason, _restore_unit_spacing,
                     _conversion_arithmetic, _conversion_of_designation, _drop_trailing_zeros, _examples_in)
from .postprocess import (_canonicalize_formatting, _canonicalize_units, _reconcile_equivalent_rows,
                          _strip_redundant_parenthetical)
from .frames import _row_context
from .normalizer import SELF_CONSISTENCY, _vote_normalize
from .parsing import set_parser_model


class GraphState(MessagesState):
    index_row_table: int
    attribute: str
    value: str
    original_value: str
    deterministic_result: str | None
    answer: str | None
    needs_review_flag: bool
    new_rows: Annotated[list[dict], add]
    data_frame: pd.DataFrame
    rules_text: str
    rule_group_names: list[str]
    row_context: str
    subject_conflict: bool
    applied: str


def start_row(state: GraphState, config: RunnableConfig) -> GraphState:

    update: dict = {}
    data_frame = state["data_frame"]
    rule_set = config["configurable"]["rule_set"]
    model = config["configurable"]["model"]

    # --- фиксируем результат ПРЕДЫДУЩЕЙ строки ---
    if state["index_row_table"] != 0:
        # needs_review — только пометка для ревьюера (жёлтая заливка), значение не подменяет
        needs_review = False
        applied: list[str] = [n for n in [state.get("applied", "")] if n]
        if state.get("deterministic_result") is not None:
            last_answer = state["deterministic_result"]
            # либо исход требует регистрации, либо код присуждён по смыслу, а не по формулировке
            needs_review = _is_pending_result(last_answer) or state.get("needs_review_flag", False)
            rules_value = last_answer  # pending/ambiguous-сообщение остаётся в ячейке
        elif state.get("answer") is not None:
            last_answer = state["answer"]
            is_ambiguous = _is_ambiguous_answer(last_answer)
            # спорную по правилам строку возвращаем как записана, но помечаем;
            # расхождение сэмплов между собой — тоже повод показать строку человеку
            # смешение предметов в найденных правилах — тоже повод показать строку человеку
            # правила чужих изделий важны, только если они что-то изменили: на нетронутой
            # записи это не «спорно», а просто «в этом документе для неё правил нет»
            changed = str(last_answer).strip() != str(state["original_value"]).strip()
            needs_review = (is_ambiguous or state.get("needs_review_flag", False)
                            or (state.get("subject_conflict", False) and changed))
            rules_value = state["original_value"] if is_ambiguous else last_answer
            if not is_ambiguous:
                rules_value = _strip_redundant_parenthetical(rules_value, state["original_value"])
                # финальная строка собирается детерминированным форматтером, если для атрибута
                # из документа извлечён жёсткий шаблон рендера (знаки после запятой + суффикс)
                template = _match_render_template(state["attribute"], rule_set.render_templates,
                                                  state.get("rule_group_names"), rules_value)
                if template:
                    rendered = _apply_render_template(rules_value, template)
                    if rendered != rules_value:
                        applied.append(f"шаблон рендера документа ({template['decimals']} зн., "
                                       f"«{template['suffix'].strip()}»)")
                    rules_value = rendered
                # проходы идут один за другим: каждый следующий видит результат предыдущего
                aligned = _align_root_spacing(rules_value, rule_set,
                                              state.get("rules_text", ""))
                if aligned != rules_value:
                    applied.append("написание корня выровнено по документу")
                rules_value = aligned
                # если документ прописал пробел, формат записи ему не указ: правило главнее
                if not _document_mandates_spacing(rule_set, state.get("rule_group_names")):
                    spaced = _restore_unit_spacing(state["original_value"], rules_value,
                                                    state["attribute"])
                    if spaced != rules_value:
                        applied.append("пробел у единицы взят из записи")
                    rules_value = spaced
                # _strip_added_words здесь больше не вызывается: она срезала слова, которые
                # документ требует дописать («NPS 4», «CL 600», «Trim 12»). Без неё MNT-DN-014
                # поднялся с 80 до 92 из 100, EMD — с 89 до 94, остальные наборы не изменились.
                guard = _rejection_reason(state["attribute"], state["original_value"],
                                          rules_value, rule_set, state.get("rules_text", ""))
                guard = guard or _conversion_of_designation(
                    state["attribute"], state["original_value"], rules_value,
                    config["configurable"].get("model"), state.get("row_context", ""))
                if guard:
                    logger.info("  [GUARD] %r -> %r: %s",
                                state["original_value"], rules_value, guard)
                    rules_value = state["original_value"]
                    needs_review = True
                    applied.append(guard)
                else:
                    arithmetic = _conversion_arithmetic(state["attribute"], state["original_value"],
                                                        rules_value)
                    if arithmetic:
                        logger.info("  [ARITH] %r -> %r: %s", state["original_value"],
                                    rules_value, arithmetic)
                        needs_review = True
                        applied.append(arithmetic)
                # документ прямо велит убирать нули в конце дробной части (карточка общих
                # положений) — после округления модель их оставляла: «50.0 kg», «5.50 m»
                if (RULE_CARDS != "off"
                        and (rule_set.global_card or {}).get("trailing_zeros") == "remove"):
                    trimmed = _drop_trailing_zeros(rules_value)
                    if trimmed != rules_value:
                        applied.append("нули в конце дробной части убраны (общее правило документа)")
                    rules_value = trimmed
                # правка разделителей, применённая к части вхождений, — повод показать строку
                partial = _partial_separator_edit(state["original_value"], rules_value)
                if partial:
                    logger.info("  [EDIT] %r -> %r: правка не по всем вхождениям (%s)",
                                state["original_value"], rules_value, partial)
                    needs_review = True
                    applied.append(f"правка разделителя не по всем вхождениям ({partial})")
            if is_ambiguous:
                applied.append("правила документа дают два разных ответа")
            elif state.get("needs_review_flag"):
                applied.append("прогоны разошлись между собой")
            if any("ответ модели не получен" in n for n in applied):
                needs_review = True  # строка осталась как записана не по правилу, а по сбою
            if not changed:
                missing = _unresolved_references(
                    rule_set, _own_rules_text(rule_set, state["attribute"],
                                              state.get("rule_group_names"),
                                              state.get("rules_text", "")))
                if missing:
                    logger.info("  [REFS] неразрешимые ссылки: %s", missing)
                    needs_review = True
                    applied.append("правило ссылается на пункт " + ", ".join(missing)
                                   + " — его текста нет в базе")
            if state.get("subject_conflict"):
                applied.append("найдены правила разных изделий, контекст строки не уточняет"
                               if changed else
                               "правила этого документа относятся к другим изделиям")
        else:
            rules_value = state["original_value"]
        # что именно изменилось — видно из самой строки, и это честнее пересказа модели
        change = _describe_change(state["original_value"], rules_value, state["attribute"])
        if change and not any("пересчёт единиц" in n for n in applied):
            applied.insert(0, change)

        # строка изменилась, а сослаться не на что: это не «правил нет», это повод посмотреть
        if not [n for n in applied if n] and _alnum(rules_value) != _alnum(state["original_value"]):
            applied.append("значение изменено, но правило не названо")
            needs_review = True

        new_row = {
            "Attribute": state["attribute"],
            "Value": state["original_value"],
            "Rules_Value": rules_value,
            "Ambiguous": needs_review,
            "Context": state.get("row_context", ""),
            "Deterministic": state.get("deterministic_result") is not None,
            "Applied": ("; ".join(dict.fromkeys(n for n in applied if n))
                        or "правила этого документа к строке не применялись"),
        }
        update["new_rows"] = [new_row]
        progress = config["configurable"].get("progress")
        if progress is not None:
            progress.step()

    # --- настраиваем ТЕКУЩУЮ строку ---
    if state["index_row_table"] < data_frame.shape[0]:
        attribute = data_frame.iloc[state["index_row_table"]]["Attribute"]
        value = data_frame.iloc[state["index_row_table"]]["Value"]

        context = _row_context(data_frame, state["index_row_table"])
        update["attribute"] = attribute
        update["original_value"] = value
        update["row_context"] = context
        update["answer"] = None
        update["needs_review_flag"] = False
        update["applied"] = ""
        update["messages"] = [RemoveMessage(id=k.id) for k in state["messages"]]

        if not _document_covers_row(rule_set, context, model):
            logger.info("=" * 80)
            logger.info("Row %d/%d | Attribute: %r | %r — документ не про это изделие, пусто",
                        state["index_row_table"] + 1, data_frame.shape[0], attribute, context)
            # пустая ячейка сама по себе говорит «документ не про это изделие» —
            # заливать её жёлтым не за чем: ревьюеру там нечего решать
            update["deterministic_result"] = ""
            update["needs_review_flag"] = False
            update["applied"] = "документ не распространяется на это изделие"
            update["value"] = ""
            update["rules_text"] = ""
            update["rule_group_names"] = []
            update["subject_conflict"] = False
            return update

        det = apply_deterministic(value, rule_set, attribute, model=model)
        if det is not None:
            det_value, det_needs_review, det_note = det
            update["applied"] = det_note
            # полностью детерминировано — общий LLM-путь не нужен
            update["deterministic_result"] = det_value
            update["needs_review_flag"] = det_needs_review
            update["value"] = det_value
            update["rules_text"] = ""
            update["rule_group_names"] = []
            update["subject_conflict"] = False
            logger.info("=" * 80)
            logger.info(
                "Row %d/%d | Attribute: %r | Value: %r | DETERMINISTIC -> %r",
                state["index_row_table"] + 1, data_frame.shape[0], attribute, value, det_value,
            )
        else:
            update["deterministic_result"] = None
            update["value"] = value
            (update["rules_text"], update["rule_group_names"],
             update["subject_conflict"]) = build_rules_text(rule_set, attribute, value, context,
                                                            model=model)
            logger.info("=" * 80)
            logger.info(
                "Row %d/%d | Attribute: %r | Value: %r | rules: %d chars",
                state["index_row_table"] + 1, data_frame.shape[0], attribute, value,
                len(update["rules_text"]),
            )

    return update


def normalize_row(state: GraphState, config: RunnableConfig) -> GraphState:
    """Ответ модели по полному набору правил, устойчивый к разбросу между прогонами."""
    model = config["configurable"]["model"]
    samples = max(1, int(config["configurable"].get("self_consistency", SELF_CONSISTENCY)))
    rule_set = config["configurable"]["rule_set"]
    answer, split, applied = _vote_normalize(model, state["attribute"], state["value"],
                                             state["rules_text"], state.get("row_context", ""),
                                             samples,
                                             _document_word_tokens(rule_set, state["attribute"]),
                                             _examples_in(rule_set, state["rules_text"]))
    logger.info("  [NORMALIZE] %r -> %r", state["value"], answer)

    # разошедшиеся сэмплы = строка, которую правила документа не решают однозначно:
    # значение берём по большинству, но помечаем строку на ручную проверку
    return {"answer": answer, "needs_review_flag": split, "applied": applied}


def next_row(state: GraphState) -> GraphState:
    return {"index_row_table": state["index_row_table"] + 1}


def condition_after_start_row(state: GraphState) -> Literal["normalize_row", "next_row", "end_node"]:
    if state["index_row_table"] >= state["data_frame"].shape[0]:
        return "end_node"
    if state.get("deterministic_result") is not None:
        return "next_row"
    return "normalize_row"


def end_node(state: GraphState) -> GraphState:
    return {}


builder = StateGraph(GraphState)


builder.add_node("start_row", start_row)


builder.add_node("normalize_row", normalize_row)


builder.add_node("next_row", next_row)


builder.add_node("end_node", end_node)


builder.add_edge(START, "start_row")


builder.add_conditional_edges("start_row", condition_after_start_row)


builder.add_edge("normalize_row", "next_row")


builder.add_edge("next_row", "start_row")


builder.add_edge("end_node", END)


graph = builder.compile()


# Колонки, которые отдаёт проход по одному документу: задаются явно, иначе таблица из
# нуля строк приходит вовсе без колонок и валит сборку результата.
ROW_COLUMNS = ["Attribute", "Value", "Rules_Value", "Ambiguous", "Context", "Deterministic",
               "Applied"]


def run_single_rule_set(data_frame: pd.DataFrame, model, rule_set, progress=None) -> pd.DataFrame:
    initial_state = {
        "messages": [],
        "index_row_table": 0,
        "data_frame": data_frame,
        "rules_text": "",
        "rule_group_names": [],
        "row_context": "",
        "subject_conflict": False,
        "new_rows": [],
        "original_value": "",
        "deterministic_result": None,
        "answer": None,
        "needs_review_flag": False,
        "applied": "",
    }
    set_parser_model(model)  # разбор значений для проверок — той же моделью, без инструментов
    final_state = graph.invoke(
        initial_state,
        config={
            "configurable": {"model": model, "rule_set": rule_set, "progress": progress},
            # на строку приходится до 3 шагов графа (start_row/normalize_row/next_row)
            "recursion_limit": 3 * data_frame.shape[0] + 25,
        },
    )
    rows = _reconcile_equivalent_rows(final_state["new_rows"], rule_set, model)
    return pd.DataFrame(_canonicalize_units(_canonicalize_formatting(rows)), columns=ROW_COLUMNS)
