# -*- coding: utf-8 -*-
"""Таблицы документа: индекс «номер таблицы → строки» для ссылок вида «see Table 12.88».

Документ может задавать форму значения не правилом с примером, а ссылкой на таблицу: в
таблице класса оборудования у атрибута написано «Canonical value of Table 12.88», а сама
таблица 12.88 стоит в другом разделе. По номеру из ссылки таблица цитируется в промпт.

Что таблица значит (реестр кодов, синонимы, пересчёт, примеры), здесь не угадывается по
подписям колонок: это решила модель при извлечении (table_roles.json, registers.load_table_roles)
— по подписям не узнать таблицу, где принятые формы стоят в первой колонке, а не в последней.

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
    """Таблица документа: номер, подпись, колонки и строки."""
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


def form_key(text: str) -> str:
    """Ключ сравнения формы: регистр и лишние пробелы не важны, всё остальное — важно."""
    return re.sub(r"\s+", " ", str(text)).strip().casefold()
