"""Агент чата: инструменты над файлами диалога и цикл вызовов.

Нормализацию агент не делает сам — он решает, ЧТО и по каким стандартам считать, и зовёт
пайплайн инструментом. Всё, что он говорит пользователю про его файлы, приходит из ответа
инструмента: выдумывать нормализованные значения ему запрещено промптом и нечем.
"""

from __future__ import annotations

import json
import logging
from typing import Callable

import pandas as pd
from langchain.tools import tool
from langchain_core.messages import ToolMessage

from .chat import AdHocRule
from .system_prompts import AGENT_PROMPT
from .workspace import StoredFile, Workspace

logger = logging.getLogger("normalization_pipeline")

# Потолок шагов цикла: агент может осмотреться, посчитать и поправить — но не крутиться вечно.
MAX_AGENT_STEPS = 8
# Сколько строк файла показывает inspect_file за один раз.
INSPECT_LIMIT = 20


def _dump(payload) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _rows_payload(stored: StoredFile, limit: int, contains: str) -> list[dict]:
    """Строки файла для осмотра, с пометкой, какие ячейки помечены на проверку."""
    frame = stored.frame
    rows = []
    for position in range(frame.shape[0]):
        row = {str(c): str(frame.iloc[position][c]) for c in frame.columns}
        if contains and contains.lower() not in " ".join(row.values()).lower():
            continue
        flagged = [col for col, flags in stored.ambiguous.items()
                   if position < len(flags) and flags[position]]
        rows.append({"row": position + 1, **row,
                     **({"flagged_columns": flagged} if flagged else {})})
        if len(rows) >= max(1, limit):
            break
    return rows


def _parse_rules(raw) -> list[AdHocRule]:
    """Правила из аргументов инструмента: [{attribute, rule}] либо просто строки."""
    rules: list[AdHocRule] = []
    for item in raw or []:
        if isinstance(item, dict):
            text = str(item.get("rule") or item.get("text") or "").strip()
            attribute = str(item.get("attribute") or "").strip()
            quantity = str(item.get("quantity") or "").strip()
        else:
            text, attribute, quantity = str(item).strip(), "", ""
        if text:
            rules.append(AdHocRule(text=text, attribute=attribute, quantity=quantity))
    return rules


def build_tools(workspace: Workspace, catalogue: list[dict], default_standards: list[str],
                normalize: Callable[[pd.DataFrame, list[str], list[AdHocRule]], tuple],
                produced: list[str],
                rule_hits: Callable[[pd.DataFrame, list[AdHocRule]], list[int]] | None = None
                ) -> list:
    """Инструменты агента, замкнутые на файлы этого диалога.

    normalize(frame, standard_names, rules) -> (таблица результата, флаги спорных ячеек).
    produced — сюда складываются файлы, созданные за этот ход: их вернём пользователю.
    rule_hits(frame, rules) -> сколько строк подошло каждому правилу пользователя.
    """

    @tool
    def list_files() -> str:
        """List the files of this conversation: what the user uploaded and what was produced.

        Returns for each file its file_id (use it in the other tools), name, kind
        ("upload" or "result"), number of rows, columns, and how many cells are flagged
        for human review.
        """
        return _dump({"files": workspace.listing()})

    @tool
    def list_standards() -> str:
        """List the standards loaded into the service, with the name to pass to normalize_file."""
        return _dump({"standards": catalogue, "selected_in_ui": default_standards})

    @tool
    def inspect_file(file_id: str, limit: int = INSPECT_LIMIT, contains: str = "") -> str:
        """Show rows of a file so you can answer questions about what is actually in it.

        `limit` caps how many rows come back; `contains` keeps only rows containing that text
        (case-insensitive), which is how you find one attribute or one value in a long table.
        For a result file, each row also says which of its columns are flagged for review.
        """
        stored = workspace.get(file_id)
        if stored is None:
            return _dump({"error": f"no file {file_id}", "files": workspace.listing()})
        return _dump({"file": stored.summary(),
                      "rows": _rows_payload(stored, limit, contains)})

    @tool
    def normalize_file(file_id: str, standards: list[str] | None = None,
                       rules: list[dict] | None = None) -> str:
        """Normalize a file against the standards and return a new result file.

        `standards` are names from list_standards; leave it empty to use what the user picked
        in the interface. `rules` are the rules the user stated in the conversation, each
        {"attribute": "<the attribute it governs, or empty>", "quantity": "<a unit, or empty>",
        "rule": "<one imperative sentence in English>"}; they are applied in their own separate
        column and only in this run. Pass every rule that should still hold, not just the newest.
        A rule about one attribute gets that attribute's name as it is written in the file (look
        it up with inspect_file). A rule about a kind of physical quantity wherever it occurs
        ("all masses in tonnes", "every temperature in degC") gets "quantity": a unit of that
        kind ("kg", "degC", "mm") and an empty "attribute": it then applies to every row whose
        value is such a quantity, whatever the attribute is called. A rule for every row gets both
        empty. The result says how many rows each rule matched.
        """
        stored = workspace.get(file_id)
        if stored is None:
            return _dump({"error": f"no file {file_id}", "files": workspace.listing()})

        missing = [c for c in ("Attribute", "Value") if c not in stored.frame.columns]
        if missing:
            return _dump({"error": f"file {stored.file_id} has no columns {missing}",
                          "columns": [str(c) for c in stored.frame.columns]})

        wanted = [str(s).strip() for s in (standards or []) if str(s).strip()] or default_standards
        known = {s["name"] for s in catalogue}
        unknown = [w for w in wanted if w not in known]
        wanted = [w for w in wanted if w in known]
        parsed = _parse_rules(rules)
        logger.info("[AGENT] normalize_file %s | стандарты %s | правил %d",
                    stored.file_id, wanted or "все", len(parsed))

        result_df, ambiguous = normalize(stored.frame, wanted, parsed)
        origin = (f"из {stored.name} по стандартам: {', '.join(wanted) if wanted else 'все'}"
                  + (f"; правил пользователя: {len(parsed)}" if parsed else ""))
        result = workspace.add(name=f"result_{stored.name}", frame=result_df, kind="result",
                               ambiguous=ambiguous, origin=origin)
        if result.file_id not in produced:
            produced.append(result.file_id)
        report = {"result": result.summary(), "unknown_standards": unknown,
                  "rows": _rows_payload(result, INSPECT_LIMIT, "")}
        if parsed and rule_hits is not None:
            report["user_rules"] = [
                {"rule": r.text, "attribute": r.attribute, "quantity": r.quantity, "rows_matched": n}
                for r, n in zip(parsed, rule_hits(stored.frame, parsed))]
        return _dump(report)

    @tool
    def edit_result(file_id: str, row: int, column: str, value: str, flag: str = "keep") -> str:
        """Change one cell of a result file, when the user explicitly asks for that correction.

        `row` is the row number as inspect_file prints it (1 = first data row), `column` a
        column name of that file, `value` the text to put there. `flag` controls the review
        mark of the cell: "keep", "clear" (the user says it is fine) or "set".
        Only a result can be edited, never a file the user uploaded. Never use this to improve
        a result on your own: it overwrites what the rules produced.
        """
        stored = workspace.get(file_id)
        if stored is None:
            return _dump({"error": f"no file {file_id}", "files": workspace.listing()})
        if stored.kind != "result":
            # правка исходника меняет то, из чего считаются все последующие прогоны, и
            # пользователь об этом не знает: свою таблицу он присылал как есть
            return _dump({"error": f"file {stored.file_id} is the uploaded {stored.name}, not a "
                                   f"result; uploads are never edited. Normalize it first, then "
                                   f"correct the result."})

        columns = {str(c).strip().lower(): c for c in stored.frame.columns}
        target = columns.get(str(column).strip().lower())
        if target is None:
            return _dump({"error": f"no column {column}",
                          "columns": [str(c) for c in stored.frame.columns]})
        position = int(row) - 1
        if not 0 <= position < stored.frame.shape[0]:
            return _dump({"error": f"row {row} is outside 1..{stored.frame.shape[0]}"})

        before = str(stored.frame.iloc[position][target])
        stored.frame.at[stored.frame.index[position], target] = value
        flags = stored.ambiguous.get(str(target))
        if flags is not None and position < len(flags) and flag in ("clear", "set"):
            flags[position] = flag == "set"
        logger.info("[AGENT] edit_result %s | строка %s, колонка %r: %r -> %r",
                    stored.file_id, row, str(target), before, value)
        if stored.file_id not in produced:
            produced.append(stored.file_id)
        return _dump({"file": stored.summary(), "changed": {"row": row, "column": str(target),
                                                            "from": before, "to": value}})

    return [list_files, list_standards, inspect_file, normalize_file, edit_result]


def run_agent(model, tools: list, history: list[dict], message: str,
              max_steps: int = MAX_AGENT_STEPS) -> str:
    """Цикл «модель ↔ инструменты»; возвращает финальный текст для пользователя."""
    by_name = {t.name: t for t in tools}
    bound = model.bind_tools(tools)
    messages: list = [{"role": "system", "content": AGENT_PROMPT}]
    messages += history
    messages.append({"role": "user", "content": message})

    last_text = ""
    for step in range(max_steps):
        try:
            response = bound.invoke(messages)
        except Exception as e:
            logger.warning("[AGENT] шаг %d не удался: %s", step + 1, e)
            return last_text or "Не удалось получить ответ модели. Попробуйте ещё раз."

        last_text = str(response.content or "").strip() or last_text
        calls = getattr(response, "tool_calls", None)
        if not calls:
            return last_text or "…"

        messages.append(response)
        for call in calls:
            tool_fn = by_name.get(call["name"])
            if tool_fn is None:
                content = _dump({"error": f"unknown tool {call['name']}"})
            else:
                try:
                    content = str(tool_fn.invoke(call["args"]))
                except Exception as e:
                    logger.warning("[AGENT] %s(%s) упал: %s", call["name"], call["args"], e)
                    content = _dump({"error": str(e)})
            logger.info("[AGENT] %s(%s) -> %s", call["name"], call["args"], content[:200])
            messages.append(ToolMessage(content=content, tool_call_id=call["id"]))

    logger.warning("[AGENT] исчерпан лимит шагов (%d)", max_steps)
    return last_text or ("Не уложился в отведённые шаги. Скажите точнее, что нужно сделать: "
                         "какой файл и какие стандарты.")
