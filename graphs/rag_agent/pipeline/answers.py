"""Разбор ответа модели и человекочитаемое «что применено» по строке."""
import json
import re

from .parsing import is_measure, parse, units_differ
from .common import _alnum


def _is_ambiguous_answer(answer) -> bool:
    """Whether the model flagged the row as ambiguous (the AMBIGUOUS token)."""
    text = str(answer).strip()
    # Strip markdown code fences, quotes and trailing punctuation models often add.
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    text = text.strip('`"\'*_#,.()[]:; ')
    return text.upper() == "AMBIGUOUS"


def _is_pending_result(answer) -> bool:
    """True, если детерминированный исход — pending/ambiguous (требует ручного решения)."""
    text = str(answer).lower()
    return PENDING_WORDING in text or AMBIGUOUS_WORDING in text


# Номер пункта стоит в начале объяснения либо после слова-указателя; «0.4 kV» в середине
# фразы — это значение, а не ссылка.
_CITED_CLAUSE_RE = re.compile(
    r"(?:^|(?:clause|per|see|under|and|,)\s+)(\d{1,2}(?:\.\d+)+)", re.I)


def _cited_clauses(payload: dict) -> str:
    """Номера пунктов, на которые сослалась модель: всё остальное она пересказывает."""
    entries = payload.get("applied")
    entries = entries if isinstance(entries, list) else [entries or ""]
    numbers: list[str] = []
    for entry in entries:
        numbers.extend(_CITED_CLAUSE_RE.findall(str(entry).strip()))
    numbers = list(dict.fromkeys(numbers))
    return "по пункту " + ", ".join(numbers[:4]) if numbers else ""


def _describe_change(value: str, answer: str, attribute: str = "") -> str:
    """Чем результат отличается от записи — по самой строке, без слов модели."""
    before, after = str(value).strip(), str(answer).strip()
    if before == after:
        return ""
    if before.replace(" ", "") == after.replace(" ", ""):
        return "убраны пробелы" if len(after) < len(before) else "добавлен пробел"
    if before.lower() == after.lower():
        return "изменён регистр"
    if before.replace(" ", "").lower() == after.replace(" ", "").lower():
        spaces = "добавлен пробел" if after.count(" ") > before.count(" ") else "убраны пробелы"
        return f"изменён регистр, {spaces}"
    if _alnum(before) == _alnum(after):
        was = "".join(dict.fromkeys(c for c in before if not c.isalnum() and not c.isspace()))
        now = "".join(dict.fromkeys(c for c in after if not c.isalnum() and not c.isspace()))
        if was != now:
            return f"разделители: «{was or 'нет'}» → «{now or 'нет'}»"
        return "изменено написание"
    was, now = parse(before, attribute), parse(after)
    if (was and now and was.single and now.single
            and is_measure(was.single.unit) and is_measure(now.single.unit)):
        if units_differ(was.single, now.single):
            return f"пересчёт: {before} → {after}"
        return f"округление: {before} → {after}"

    before_words = sorted(re.findall(r"\w+", before.lower()))
    after_words = sorted(re.findall(r"\w+", after.lower()))
    if before_words == after_words:
        return "переставлены части значения"
    dropped = [w for w in dict.fromkeys(re.findall(r"\w+", before)) if w.lower() not in after_words]
    added = [w for w in dict.fromkeys(re.findall(r"\w+", after)) if w.lower() not in before_words]
    parts = []
    if dropped:
        parts.append("убрано: " + ", ".join(f"«{w}»" for w in dropped[:3]))
    if added:
        parts.append("добавлено: " + ", ".join(f"«{w}»" for w in added[:3]))
    return "; ".join(parts) or f"значение заменено на «{after}»"


def _parse_payload(answer) -> dict:
    """Ответ модели как объект: кроме result там лежит и объяснение, почему так."""
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", str(answer).strip()).strip()
    try:
        obj = json.loads(text)
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


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


# Формулировки исходов (легко локализовать/поменять)
PENDING_WORDING = "pending registration"


AMBIGUOUS_WORDING = "ambiguous match"
