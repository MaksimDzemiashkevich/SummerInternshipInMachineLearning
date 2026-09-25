# -*- coding: utf-8 -*-
"""Таблицы документа: индекс «номер таблицы → строки» и таблицы синонимов.

Документ может задавать форму значения не правилом с примером, а ссылкой на таблицу: в
таблице класса оборудования у атрибута написано «Canonical value of Table 12.88», а сама
таблица 12.88 («Yes ← Y; True; 1») стоит в другом разделе. Раньше такая ссылка не
разрешалась (разрешались только номера пунктов), а таблица «каноническое значение ←
принятые формы» читалась как таблица кодов «фраза → код» и притом задом наперёд:
«SCH STD ← Standard; STD» превращалось в реестр {'SCH STD': 'STD'}.

Таблицы берутся прямо со страниц (data_after_check.json): там у них сохранены колонки и
строки, а в группах правил они уже разложены строками текста и частично разнесены по группам.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from graphs.structured_documents.block_grouping import parse_page_content

# «Table 12.41 — …», «Table B.3 — …», «Table 6.1 …»
_TABLE_ID_RE = re.compile(r"^\s*Table\s+([A-Z](?:\.\d+)+|\d+(?:\.\d+)*)\b", re.I)
# ссылки в тексте: «Table 12.88», «Tables 11.1, 11.2 and 11.3»
_TABLE_REF_RE = re.compile(
    r"\bTables?\s+((?:[A-Z](?:\.\d+)+|\d+(?:\.\d+)*)(?:\s*(?:,|and|or)\s*(?:[A-Z](?:\.\d+)+|\d+(?:\.\d+)*))*)")
_TABLE_ID_IN_REF_RE = re.compile(r"[A-Z](?:\.\d+)+|\d+(?:\.\d+)*")
# Длиннее — уже не форма значения, а фраза описания
MAX_FORM_CHARS = 60


@dataclass
class DocTable:
    id: str
    title: str
    columns: list[str]
    rows: list[dict] = field(default_factory=list)

    def lines(self) -> list[str]:
        """Строки таблицы в той же форме, что и в группах правил («колонка: 'значение'; …»)."""
        head = f"{self.title} (columns: {'; '.join(self.columns)})"
        return [head] + ["; ".join(f"{k}: '{v}'" for k, v in row.items() if str(v).strip())
                         for row in self.rows]


def table_references(text: str) -> list[str]:
    """Номера таблиц, на которые ссылается текст, в порядке появления."""
    ids: list[str] = []
    for match in _TABLE_REF_RE.finditer(str(text)):
        ids.extend(_TABLE_ID_IN_REF_RE.findall(match.group(1)))
    return list(dict.fromkeys(ids))


def load_tables(doc_dir: Path) -> dict[str, DocTable]:
    """Номер таблицы → таблица, по тексту страниц документа.

    Таблица без заголовка с теми же колонками, что у предыдущей, — её продолжение на
    следующей странице (заголовок повторяют не все документы).
    """
    path = doc_dir / "data_after_check.json"
    if not path.exists():
        return {}
    try:
        pages = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    tables: dict[str, DocTable] = {}
    last: DocTable | None = None
    for page_no, page in enumerate(pages):
        content = parse_page_content(page.get("content")) if isinstance(page, dict) else None
        for block_no, block in enumerate((content or {}).get("blocks", [])):
            if block.get("type") != "table":
                continue
            title = str(block.get("title") or "").strip()
            columns = [str(c) for c in block.get("columns") or []]
            rows = [r for r in block.get("rows") or []
                    if isinstance(r, dict) and any(str(v).strip() for v in r.values())]
            match = _TABLE_ID_RE.match(title)
            if match:
                last = tables.setdefault(match.group(1), DocTable(match.group(1), title, columns))
                last.rows.extend(rows)
            elif not title and last is not None and columns == last.columns:
                last.rows.extend(rows)
            elif not title and rows:
                # заголовок потерян при распознавании: сослаться на такую таблицу нельзя, но
                # её строки лежат в группе правил, и по ним она находит своего владельца
                last = DocTable(f"p{page_no}.{block_no}", "", columns, list(rows))
                tables[last.id] = last
            else:
                last = None
    return tables


def _forms(cell: str) -> list[str]:
    return [f.strip() for f in str(cell).split(";") if f.strip()]


# Подпись колонки, в которой перечислены другие написания того же значения. Это словарь
# табличной лексики вообще, а не слова конкретного документа.
_FORMS_HEADER_RE = re.compile(
    r"\b(?:accepted|source forms?|synonyms?|aliases|alternative|variants?|also written|other forms?)\b",
    re.I)


def synonym_map(table: DocTable) -> dict[str, str]:
    """Принятая форма → каноническое значение, если таблица — таблица синонимов; иначе {}.

    Колонка форм — ПОСЛЕДНЯЯ, её подпись называет другие написания («Accepted source forms»),
    и перечни через «;» есть только в ней. Каноническое значение — первая колонка (колонки
    между ними — пояснения). Одного «;» мало: в таблице-указателе атрибутов («Written form(s):
    Canonical value of Table 12.32; Canonical value of Table 9.7») перечень тоже стоит в
    последней колонке. Таблица «описание | код» без такой подписи сюда не попадает.
    Бывает, что распознавание кладёт каноническое значение ключом строки ({'Yes': 'Y; True'}),
    — тогда ключ, которого нет среди колонок, и есть каноническое значение.
    """
    columns = table.columns
    if len(columns) < 2 or not _FORMS_HEADER_RE.search(columns[-1]):
        return {}
    listed = {c for row in table.rows for c, v in row.items() if c in columns and ";" in str(v)}
    if listed - {columns[-1]}:
        return {}
    pairs: list[tuple[str, str]] = []
    for row in table.rows:
        if all(k in columns for k in row):
            canonical = str(row.get(columns[0], "")).strip()
            pairs.extend((form, canonical) for form in _forms(row.get(columns[-1], "")))
        else:
            for key, cell in row.items():
                if key not in columns:
                    pairs.extend((form, str(key).strip()) for form in _forms(cell))
                    pairs.append((str(key).strip(), str(key).strip()))
    mapping: dict[str, str] = {}
    clashing: set[str] = set()
    for form, canonical in pairs:
        if (not canonical or ";" in canonical or len(form) > MAX_FORM_CHARS
                or len(canonical) > MAX_FORM_CHARS):
            continue
        for key in (form_key(form), form_key(canonical)):
            if mapping.setdefault(key, canonical) != canonical and key != form_key(mapping[key]):
                clashing.add(key)
    # форма, ведущая к двум каноническим значениям («Canonical value only» в каждой строке),
    # — не соответствие, а заглушка или ошибка распознавания
    return {k: v for k, v in mapping.items() if k not in clashing}


def form_key(text: str) -> str:
    """Ключ сравнения формы: регистр и лишние пробелы не важны, всё остальное — важно."""
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def synonym_owners(groups: list[dict], tables: dict[str, DocTable],
                   synonyms: dict[str, dict[str, str]]) -> tuple[dict[str, list[str]], list[str]]:
    """(атрибут → номера таблиц синонимов, таблицы, действующие для всех атрибутов).

    Атрибут получает таблицу, если: строки таблицы лежат в его группе правил; правило его
    группы ссылается на неё; или строка таблицы атрибутов называет его и ссылается на неё
    («Attribute: 'Roof type'; Written form: 'Canonical value of Table 12.41'»). Таблица,
    строки которой лежат в global-группе, действует для любого атрибута (значения «N/A», «TBA»).
    """
    owners: dict[str, list[str]] = {}
    universal: list[str] = []

    def add(attribute: str, table_id: str) -> None:
        key = str(attribute).strip().lower()
        if key and table_id not in owners.setdefault(key, []):
            owners[key].append(table_id)

    for group in groups:
        body = "\n".join(str(r) for r in group.get("rules", []))
        held = [tid for tid in synonyms
                if (tables[tid].title in body if tables[tid].title
                    else any(line in body for line in tables[tid].lines()[1:]))]
        if group.get("type") == "global":
            universal.extend(t for t in held if t not in universal)
            continue
        for tid in held + [t for t in table_references(body) if t in synonyms]:
            add(group.get("attribute", ""), tid)

    for table in tables.values():
        attribute_column = next((c for c in table.columns if c.strip().lower() == "attribute"), None)
        if not attribute_column:
            continue
        for row in table.rows:
            if attribute_column in row:
                refs = [t for v in row.values() for t in table_references(v) if t in synonyms]
                for tid in refs:
                    add(row[attribute_column], tid)
            else:
                # имя атрибута распознано ключом строки: {'Sensor type': 'Canonical value of Table 12.46'}
                for key, cell in row.items():
                    if key not in table.columns:
                        for tid in (t for t in table_references(cell) if t in synonyms):
                            add(key, tid)
    return owners, universal
