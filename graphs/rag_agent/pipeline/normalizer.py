"""Вызов модели на строку: цикл инструментов, нормализация с проверкой, голосование сэмплов."""
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import ToolMessage

from ..modules.system_prompts import SYSTEM_PROMPT1, VERIFY_PROMPT
from ..modules.tools import unit_mismatch, array_of_tools
from .common import _context_line, logger
from .answers import _cited_clauses, _is_ambiguous_answer, _parse_payload, _parse_result
from .guards import _example_leak


TOOLS_BY_NAME = {t.name: t for t in array_of_tools}


# Сколько независимых сэмплов усредняется голосованием на LLM-пути (1 — выключено).
# Больше сэмплов — устойчивее ответ на спорных строках, но дороже прогон.
SELF_CONSISTENCY = int(os.getenv("NORMALIZER_SAMPLES", "1"))


def _tool_note(name: str, args: dict, result: str) -> str:
    """Что именно сделал инструмент — словами, по его же аргументам и ответу.

    Вызов, ничего не изменивший (замена строки на саму себя, пересчёт в ту же единицу),
    в объяснение не идёт: это не применённое правило, а холостой ход.
    """
    if str(result).startswith("cannot convert"):
        return f"пересчёт не выполнен: {result}"
    if name == "convert_units":
        source = f"{args.get('value', '')} {args.get('from_unit', '')}".strip()
        target = str(result)
        if not unit_mismatch(str(args.get("from_unit", "")), str(args.get("to_unit", ""))):
            return "" if source.strip() == target.strip() else f"округление: «{source}» → {target}"
        return f"пересчёт единиц: «{source}» → {target}"
    if name == "remove_characters":
        return (f"удалены символы «{args.get('characters')}» из «{args.get('text')}»"
                if str(args.get("characters", "")) else "")
    if name == "replace_substring":
        if str(args.get("find")) == str(args.get("replace_with")):
            return ""
        return (f"замена «{args.get('find')}» на «{args.get('replace_with')}» "
                f"в «{args.get('text')}»")
    return f"{name}: {result}"


# Потолок шагов цикла инструментов на один сэмпл. Штатной строке хватает одного-двух
# вызовов; без потолка модель, зациклившаяся на холостой замене, молотит бесконечно.
MAX_TOOL_STEPS = 6


_REPEATED_CALL_REPLY = ("this exact call was already made and returned the same result; "
                        "do not call it again — answer now with the JSON object")


_BUDGET_REPLY = ("no more tool calls are available for this value; answer now with the "
                 "JSON object, using what the previous calls returned")


def _run_tool_call(tc: dict, notes: list[str]) -> str:
    """Выполняет один вызов и копит заметку о сделанном."""
    tool_fn = TOOLS_BY_NAME.get(tc["name"])
    if tool_fn is None:
        return "unknown tool"
    try:
        content = str(tool_fn.invoke(tc["args"]))
    except Exception as e:
        return f"error: {e}"
    logger.info("  [TOOL] %s(%s) -> %r", tc["name"], tc["args"], content)
    note = _tool_note(tc["name"], tc["args"], content)
    if note:
        notes.append(note)
    return content


def _invoke_with_tools(model, messages) -> tuple:
    """Инвокает модель, прогоняя цикл tool-calls. Возвращает (ответ, что сделали инструменты).

    Повторный вызов с теми же аргументами не выполняется, а число шагов ограничено: иначе
    модель, которой инструмент возвращает то же самое, вызывает его снова и снова.
    """
    notes: list[str] = []
    seen: set = set()
    response = model.invoke(messages)

    for _ in range(MAX_TOOL_STEPS):
        calls = getattr(response, "tool_calls", None)
        if not calls:
            return response, notes
        tool_messages = []
        for tc in calls:
            key = (tc["name"], json.dumps(tc["args"], sort_keys=True, default=str))
            if key in seen:
                logger.info("  [TOOL] повтор %s(%s) — не выполняем", tc["name"], tc["args"])
                content = _REPEATED_CALL_REPLY
            else:
                seen.add(key)
                content = _run_tool_call(tc, notes)
            tool_messages.append(ToolMessage(content=content, tool_call_id=tc["id"]))
        messages = messages + [response] + tool_messages
        response = model.invoke(messages)

    if getattr(response, "tool_calls", None):
        logger.warning("  [TOOL] исчерпан лимит вызовов (%d), просим ответить без инструментов",
                       MAX_TOOL_STEPS)
        closing = [ToolMessage(content=_BUDGET_REPLY, tool_call_id=tc["id"])
                   for tc in response.tool_calls]
        messages = messages + [response] + closing
        response = model.invoke(messages)
    return response, notes


def _normalize_once(model, attribute: str, value: str, rules_text: str,
                    context: str = "") -> tuple[str, str]:
    """Один сэмпл: применение правил + проверка кандидата на overreach.

    Правила подаются целиком за раз, а не порциями: при инкрементальной подаче ранний
    (возможно нерелевантный) чанк закрепляет ответ, и последующие правила уже только
    дополняют его, не пересматривая.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT1},
        {"role": "user", "content": (f"Attribute: {attribute}\n{_context_line(context)}"
                                     f"Value: {value}\n\n"
                                     f"Rules:\n{rules_text}")},
    ]
    response, tool_notes = _invoke_with_tools(model, messages)
    payload = _parse_payload(response.content)
    candidate = _parse_result(json.dumps(payload) if payload else "").strip()
    applied = "; ".join(x for x in (*tool_notes, _cited_clauses(payload)) if x)
    if not candidate:
        # модель не дала разбираемого ответа (например, упёрлась в лимит вызовов):
        # запись остаётся как есть, а строка получает пометку
        logger.warning("  [NORMALIZE] ответа нет, оставляем запись %r", value)
        return str(value), "; ".join(x for x in (applied, "ответ модели не получен") if x)
    if not candidate or _is_ambiguous_answer(candidate):
        return candidate, applied

    # второй проход: precondition, исключения, утечка из worked-example, полнота
    messages = [
        {"role": "system", "content": VERIFY_PROMPT},
        {"role": "user", "content": (f"Attribute: {attribute}\n{_context_line(context)}"
                                     f"Original value: {value}\n"
                                     f"Candidate answer: {candidate}\n\n"
                                     f"Rules:\n{rules_text}")},
    ]
    review_response, review_tools = _invoke_with_tools(model, messages)
    review = _parse_payload(review_response.content)
    checked = str(review.get("result", "")).strip()
    if checked and checked != candidate:
        logger.info("  [VERIFY] %r -> %r", candidate, checked)
        extra = "; ".join(x for x in (*review_tools, _cited_clauses(review)) if x)
        applied = "; ".join(x for x in (applied, extra) if x)
    return (checked or candidate), applied


def _vote_normalize(model, attribute: str, value: str, rules_text: str,
                    context: str, samples: int, allowed: set = frozenset(),
                    examples: list[tuple[str, str]] | None = None) -> tuple[str, bool, str]:
    """Несколько независимых сэмплов и большинство; второй элемент — разошлись ли они.

    Спорные строки модель трактует по-разному от запуска к запуску, поэтому берётся
    несколько сэмплов: это и повышает точность, и делает результат воспроизводимым
    (одна и та же таблица даёт одну и ту же выдачу).
    """
    args = (model, attribute, value, rules_text, context)
    if samples <= 1:
        sampled = [_normalize_once(*args)]
    else:
        with ThreadPoolExecutor(max_workers=samples) as pool:
            sampled = list(pool.map(lambda _: _normalize_once(*args), range(samples)))

    # сэмпл, перенёсший слово из записи-примера, не голосует: иначе при двух таких из трёх
    # побеждал чужой ответ. Если перенесли все, решает guard после голосования
    leaks = [_example_leak(value, v, rules_text, allowed, examples) for v, _ in sampled]
    clean = [s for s, leak in zip(sampled, leaks) if not leak]
    if clean and len(clean) < len(sampled):
        logger.info("  [LEAK] отброшено сэмплов со словами из примеров: %d", len(sampled) - len(clean))
        sampled = clean
    elif not clean and any(leaks):
        # перенесли все: ещё одна попытка, где названо, какое слово чужое. Удалась — ответ
        # берём, но строку помечаем: сама модель с первого раза этого не увидела
        borrowed = sorted(set().union(*map(set, leaks)))
        note = ("\n\nNote on earlier attempts: they put " + ", ".join(f'"{w}"' for w in borrowed)
                + " into the answer. That comes from a worked example of the document, not from "
                  "this value. Apply the rule's mechanism to this value's own words only.")
        retry, retry_applied = _normalize_once(model, attribute, value, rules_text + note, context)
        if not _example_leak(value, retry, rules_text, allowed, examples):
            logger.info("  [LEAK] все сэмплы взяли %s из примеров, повтор дал %r", borrowed, retry)
            return retry, True, retry_applied

    # большинство; при равенстве — вариант, встретившийся раньше (стабильный tie-break)
    candidates = [value for value, _ in sampled]
    counts = Counter(candidates)
    answer = max(candidates, key=lambda c: (counts[c], -candidates.index(c)))
    applied = next((note for value, note in sampled if value == answer), "")
    if len(counts) > 1:
        logger.info("  [VOTE] %r -> %r", dict(counts), answer)
    return answer, len(counts) > 1, applied
