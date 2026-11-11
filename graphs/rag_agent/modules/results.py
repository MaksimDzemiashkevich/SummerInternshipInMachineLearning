"""Файл результата как данные: какие колонки чьи, уровень уверенности строки, сводки по проблемам.

Всё здесь считает код (pandas), а не модель: агент только выбирает, какой вопрос задать. Ответ
«у какого изделия больше всего ошибок» должен быть посчитан, а не угадан по первым строкам.
"""

from __future__ import annotations

import re

import pandas as pd

# Суффикс колонки-объяснения (тот же, что в pipeline.user_rules — продублирован, чтобы этот
# модуль не тянул за собой весь конвейер).
APPLIED_SUFFIX = " — применено"
USER_RULES_COLUMN = "Мои правила"

# Что считать «проблемой» строки.
#   flagged         — ячейка жёлтая: сервис не уверен или документ велит проверить;
#   changed         — исходная запись не соответствовала стандарту: результат отличается от неё;
#   low_confidence  — ответ не подтверждён проверенным кодом фактом (уровень ниже «высокой»).
METRICS = ("flagged", "changed", "low_confidence")

# Уровень строки — по первым словам пояснения, которое пишет spec_runner.
_LEVEL_PREFIXES = (
    ("высокая", "high"),
    ("средняя", "medium"),
    ("низкая", "low"),
    ("разошлись", "disagreed"),
    ("нет в списке", "not_in_list"),
    ("документ велит", "document_flag"),
    ("документ не распространяется", "not_covered"),
    ("ответ не получен", "no_answer"),
    ("принято предложение", "accepted"),
    ("исправлено вручную", "edited"),
)
_UNSURE_LEVELS = {"medium", "low", "disagreed", "not_in_list", "no_answer"}

_QUOTED_RE = re.compile(r"«([^«»]*)»")


def level_of(note: str) -> str:
    text = str(note or "").strip().lower()
    for prefix, level in _LEVEL_PREFIXES:
        if text.startswith(prefix):
            return level
    return "other"  # пояснение прежнего графа: уровней у него нет


def proposals_of(note: str, cell: str, level: str) -> list[str]:
    """Что сервис предлагал записать в жёлтую ячейку (по пояснению spec_runner).

    «низкая: предложено «X»» — одно предложение, в ячейке осталась запись; «разошлись: «A» /
    «B»» — два; «в записи нет единицы, предложено в единице документа» — предложение уже стоит в
    ячейке. Остальное предложений не несёт.
    """
    text = str(note or "")
    if level == "low" and "предложено «" in text:
        return _QUOTED_RE.findall(text.split("предложено", 1)[1])[:1]
    if level == "low" and "предложено в единице документа" in text:
        return [str(cell)]
    if level == "disagreed":
        # «UNMATCHED» — ответ «такого значения в списке нет», а не значение, которое можно записать
        return [q for q in _QUOTED_RE.findall(text)[:2] if q and q not in ("—", "UNMATCHED")]
    return []


def _same(left, right) -> bool:
    return " ".join(str(left).split()) == " ".join(str(right).split())


# ---------------------------------------------------------------------- разметка колонок
def layout_of(frame: pd.DataFrame, known_standards: set[str] | None = None) -> dict:
    """Разметка таблицы, у которой её не записали (результат, присланный обратно файлом).

    Колонка стандарта узнаётся по соседу «<имя> — применено»; всё до первой такой колонки —
    исходные колонки. Без колонок-объяснений таблица считается исходной.
    """
    columns = [str(c) for c in frame.columns]
    standards, user_rules = [], None
    for column in columns:
        if not column.endswith(APPLIED_SUFFIX):
            continue
        value_column = column[: -len(APPLIED_SUFFIX)]
        if value_column not in columns:
            continue
        base = re.sub(r" \(\d+\)$", "", value_column)
        entry = {"column": value_column, "applied": column}
        if base == USER_RULES_COLUMN:
            user_rules = entry
        else:
            entry["standard"] = base if not known_standards or base in known_standards else value_column
            standards.append(entry)
    produced = {e["column"] for e in standards} | {e["applied"] for e in standards}
    if user_rules:
        produced |= {user_rules["column"], user_rules["applied"]}
    source = [c for c in columns if c not in produced]
    return {"source_columns": source,
            "context_columns": [c for c in source if c not in ("Attribute", "Value")],
            "standards": standards, "user_rules": user_rules}


def _column(frame: pd.DataFrame, name: str) -> str | None:
    """Колонка по имени без учёта регистра и пробелов по краям («item» → «Item»)."""
    wanted = str(name).strip().lower()
    return next((str(c) for c in frame.columns if str(c).strip().lower() == wanted), None)


# ---------------------------------------------------------------------- строки результата
def result_rows(frame: pd.DataFrame, layout: dict, ambiguous: dict[str, list[bool]],
                standard: str | None = None) -> list[dict]:
    """Одна запись на (строка таблицы, стандарт): запись, результат, пометка, уровень, предложения."""
    rows = []
    entries = [e for e in layout["standards"]
               if not standard or standard.lower() in (e["standard"].lower(), e["column"].lower())]
    for position in range(frame.shape[0]):
        source = frame.iloc[position]
        record = str(source.get("Value", ""))
        context = {c: str(source[c]) for c in layout["context_columns"]
                   if pd.notna(source[c]) and str(source[c]).strip()}
        for entry in entries:
            cell = source[entry["column"]]
            cell = "" if pd.isna(cell) else str(cell)
            note = source.get(entry["applied"], "")
            note = "" if pd.isna(note) else str(note)
            flags = ambiguous.get(entry["column"]) or []
            flagged = bool(flags[position]) if position < len(flags) else False
            level = level_of(note)
            covered = level != "not_covered"
            rows.append({
                "row": position + 1, "Attribute": str(source.get("Attribute", "")), "Value": record,
                **context, "standard": entry["standard"], "result": cell, "flagged": flagged,
                "level": level, "note": note,
                "changed": covered and cell != "" and not _same(cell, record),
                "low_confidence": covered and (level in _UNSURE_LEVELS or (level == "other" and flagged)),
                "proposals": proposals_of(note, cell, level) if flagged else [],
            })
    return rows


def _matches(row: dict, filters: dict | None) -> bool:
    """Фильтр «колонка → подстрока» без учёта регистра; ключ — любое поле записи."""
    for key, wanted in (filters or {}).items():
        field = next((k for k in row if k.lower() == str(key).strip().lower()), None)
        # список — «любое из»: строки про мощность — это и «Rated power», и «Compressor input power»
        options = wanted if isinstance(wanted, (list, tuple)) else [wanted]
        if field is None or not any(str(o).strip().lower() in str(row[field]).lower() for o in options):
            return False
    return True


def overview(frame: pd.DataFrame, layout: dict, ambiguous: dict[str, list[bool]]) -> dict:
    """Сводка файла: роли колонок, сколько чего, проблемы по стандартам и уровни уверенности."""
    rows = result_rows(frame, layout, ambiguous)
    per_standard = []
    for entry in layout["standards"]:
        mine = [r for r in rows if r["standard"] == entry["standard"]]
        levels: dict[str, int] = {}
        for r in mine:
            levels[r["level"]] = levels.get(r["level"], 0) + 1
        per_standard.append({
            "standard": entry["standard"], "column": entry["column"],
            **{m: sum(1 for r in mine if r[m]) for m in METRICS},
            "not_covered": levels.get("not_covered", 0), "levels": levels})
    distinct = {c: int(frame[c].nunique()) for c in layout["context_columns"] + ["Attribute"]
                if c in frame.columns}
    return {"rows": int(frame.shape[0]), "source_columns": layout["source_columns"],
            "context_columns": layout["context_columns"], "distinct_values": distinct,
            "standards": per_standard,
            "user_rules_column": (layout["user_rules"] or {}).get("column")}


def aggregate(frame: pd.DataFrame, layout: dict, ambiguous: dict[str, list[bool]],
              group_by: list[str], metric: str = "flagged", standard: str | None = None,
              top: int = 10, filters: dict | None = None) -> dict:
    """Топ групп по числу проблемных строк: «у какого Item больше всего жёлтых», «топ-5 атрибутов».

    Группировка — по любым колонкам исходника (контекст, Attribute) и по «standard» / «level».
    Доля считается от всех строк группы, чтобы большой item не выглядел худшим только от размера.
    """
    if metric not in METRICS:
        return {"error": f"metric must be one of {list(METRICS)}"}
    rows = [r for r in result_rows(frame, layout, ambiguous, standard) if _matches(r, filters)]
    keys = []
    for name in group_by or ["Attribute"]:
        key = next((k for k in (rows[0] if rows else {}) if k.lower() == str(name).strip().lower()), None)
        if key is None:
            return {"error": f"no column {name}",
                    "group_by_options": layout["context_columns"] + ["Attribute", "standard", "level"]}
        keys.append(key)
    groups: dict[tuple, dict] = {}
    for r in rows:
        # пустая ячейка контекста в запись строки не попадает (result_rows): такие строки — своя
        # группа «», а не KeyError, ронявший /chat на файле с незаполненным Item
        group = groups.setdefault(tuple(r.get(k, "") for k in keys), {"total": 0, "count": 0, "examples": []})
        group["total"] += 1
        if r[metric]:
            group["count"] += 1
            if len(group["examples"]) < 3:
                group["examples"].append({"row": r["row"], "Value": r["Value"], "result": r["result"],
                                          "note": r["note"][:160]})
    ranked = sorted(((k, g) for k, g in groups.items() if g["count"]),
                    key=lambda kg: (-kg[1]["count"], -kg[1]["count"] / kg[1]["total"]))
    return {"metric": metric, "standard": standard or "all", "group_by": keys,
            "groups_total": len(groups), "groups_with_issues": len(ranked),
            "top": [{**dict(zip(keys, k)), "count": g["count"], "of_rows": g["total"],
                     "share": round(g["count"] / g["total"], 2), "examples": g["examples"]}
                    for k, g in ranked[:max(1, int(top))]]}


def issues(frame: pd.DataFrame, layout: dict, ambiguous: dict[str, list[bool]],
           metric: str = "flagged", standard: str | None = None, filters: dict | None = None,
           limit: int = 30) -> dict:
    """Проблемные строки с тем, что с ними не так: запись, результат, уровень, причина, предложения."""
    if metric not in METRICS:
        return {"error": f"metric must be one of {list(METRICS)}"}
    found = [r for r in result_rows(frame, layout, ambiguous, standard)
             if r[metric] and _matches(r, filters)]
    shown = [{k: v for k, v in r.items() if k not in METRICS or k == metric} for r in found[:max(1, int(limit))]]
    return {"metric": metric, "standard": standard or "all", "found": len(found),
            "shown": len(shown), "rows": shown}
