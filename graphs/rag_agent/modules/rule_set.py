"""Документ-стандарт в памяти сервиса: его группы правил, общие положения и данные извлечения.

RuleSet собирается из папки документа в DOCUMENTS_OUTPUT (rule_groups.json и соседние файлы).
Путь по паспортам берёт из него имя, общие положения, разделы классов изделий и начало документа;
остальные поля (реестры, синонимы, шаблоны, поисковые индексы) нужны только запасному пути.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from graphs.structured_documents.block_grouping import parse_page_content

from .registers import legacy_fields, read_json
from .tables import DocTable, load_tables

logger = logging.getLogger(__name__)


@dataclass
class GroupEntry:
    """Одна группа правил: имя атрибута, алиасы, текст правил и родительская тема."""
    attribute: str
    aliases: list[str]
    body: str
    parent: str = ""
    shapes: set[str] = field(default_factory=set)


@dataclass
class RuleSet:
    """Правила одного документа и всё, что из них построено при загрузке."""
    name: str
    global_rules: str
    groups: list[GroupEntry]
    deterministic: dict[str, str] = field(default_factory=dict)
    phrase_map: dict[str, str] = field(default_factory=dict)
    closed_registers: dict[str, dict[str, str]] = field(default_factory=dict)
    attribute_to_register: dict[str, list[str]] = field(default_factory=dict)
    number_first: bool = False
    deterministic_order: dict[str, bool] = field(default_factory=dict)
    deterministic_attributes: dict[str, str] = field(default_factory=dict)
    excluded_attributes: list[str] = field(default_factory=list)
    render_templates: dict[str, dict] = field(default_factory=dict)
    parent_to_indices: dict[str, list[int]] = field(default_factory=dict)
    clause_index: dict[str, str] = field(default_factory=dict)
    group_subjects: dict[str, str] = field(default_factory=dict)
    # таблицы документа по номеру и перечни синонимов «принятая форма → запись»
    tables: dict[str, DocTable] = field(default_factory=dict)
    synonyms: dict[str, dict[str, str]] = field(default_factory=dict)
    synonym_owners: dict[str, list[str]] = field(default_factory=dict)
    universal_synonyms: list[str] = field(default_factory=list)
    # класс изделия → его правила (определение, таблица атрибутов, пункты раздела класса)
    class_blocks: dict[str, str] = field(default_factory=dict)
    # карточки правил (rule_cards.json) и принятые формы из них (группа → форма → запись)
    cards: dict[str, dict] = field(default_factory=dict)
    global_card: dict = field(default_factory=dict)
    card_forms: dict[str, dict[str, str]] = field(default_factory=dict)
    # примеры «было → стало» из карточек: [{"from", "to", "group"}]
    examples: list[dict] = field(default_factory=list)
    # начало документа (заголовок, тема, вводная часть): по нему решается, про какое оборудование документ
    opening_text: str = ""
    # поисковые индексы запасного пути строятся при первом поиске (search_index);
    # index_source — откуда их брать: модель эмбеддингов и папка кэша индексов
    index_source: Any = field(default=None, repr=False, compare=False)
    store: Any = field(default=None, repr=False, compare=False)
    bm25: Any = field(default=None, repr=False, compare=False)
    _index_lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def search_index(self):
        """(семантический индекс, BM25) этого документа; при первом вызове строятся."""
        from rank_bm25 import BM25Okapi

        from .search import load_vector_store, search_text, tokenize
        if self.index_source is None:
            raise RuntimeError(f"документ {self.name}: не задано, где строить поисковый индекс (index_source)")
        with self._index_lock:
            if self.store is None:
                self.store = load_vector_store(self.index_source.embedding_model, self.index_source.vector_store_dir,
                                               self.groups, self.name)
                self.bm25 = BM25Okapi([tokenize(search_text(e)) for e in self.groups])
        return self.store, self.bm25


# Все загруженные документы по обозначению («OGXSTD410»): правило одного документа может
# ссылаться на пункт другого («per 6.2 of OGX-STD 410»), и текст пункта берётся отсюда.
RULE_SET_LIBRARY: dict[str, RuleSet] = {}

# группы о служебных частях документа (выходные данные, оглавление, юридический текст), а не о значениях
_NON_RULE_ATTRIBUTE_MARKERS = ("metadata", "copyright", "legal", "liability", "disclaimer", "table of contents",
                               "bibliography", "foreword", "acknowledgment", "contact information",
                               "distribution", "publisher", "provided by")
_SHAPE_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
_CLAUSE_DEF_RE = re.compile(r"^\s*(\d+(?:\.\d+)+)\s")  # строка, начинающаяся с номера пункта
_ATTACHED_RE = re.compile(r"^\s*(?:EXAMPLE|NOTE)\b", re.I)
OPENING_CHARS = 2500


def value_shape(text: str) -> str:
    """Форма числового значения: числа → «9», без пробелов и регистра.

    «600 x 600mm» и «500 x 1200mm» — одна форма «9x9mm», «2.5 bar» и «1.5 bar» — «9bar». У значения
    без цифр формы нет: по словам совпадает слишком многое.
    """
    text = str(text).strip().lower()
    if not re.search(r"\d", text):
        return ""
    return re.sub(r"\s+", "", _SHAPE_NUMBER_RE.sub("9", text))


def document_id(text: str) -> str:
    """Обозначение документа без разделителей и регистра: «OGX-STD 410» → «OGXSTD410».

    Из имени папки берутся ведущие части до первого слова в нижнем регистре:
    «OGX-STD-410_multi_equipment_…» → «OGXSTD410», «XYZ_STD_100_normalization_rules» → «XYZSTD100».
    """
    parts = []
    for token in re.split(r"_", str(text)):
        if token and token == token.lower() and not token.isdigit():
            break
        parts.append(token)
    return re.sub(r"[^A-Z0-9]", "", "".join(parts or [str(text)]).upper())


def split_groups(groups: list[dict]) -> tuple[str, list[GroupEntry]]:
    """(текст общих положений, группы атрибутов) документа.

    Группы классов изделий (kind="class") идут отдельно (class_blocks); группы о служебных частях
    документа и правила идентичности изделия (kind="identity") в группы атрибутов не входят:
    они называют атрибуты, но ничего не говорят о том, как записать значение.
    """
    global_rules: list[str] = []
    entries: list[GroupEntry] = []
    for group in groups:
        header = group.get("attribute", "").strip()
        body = "\n".join(rule.strip() for rule in group.get("rules", []))
        if group.get("type") == "global":
            global_rules.append(body)
        elif group.get("kind") in ("class", "identity"):
            continue
        elif not any(marker in header.lower() for marker in _NON_RULE_ATTRIBUTE_MARKERS):
            aliases = [a.strip() for a in group.get("aliases", []) if a and a.strip()]
            entries.append(GroupEntry(attribute=header, aliases=aliases, body=body,
                                      parent=group.get("parent", "") or header))
    return "\n".join(global_rules), entries


def class_blocks(groups: list[dict]) -> dict[str, str]:
    """Название класса изделия → его правила (группы kind="class").

    Документ на много классов оборудования описывает каждый класс своим разделом: что это, какие у
    него атрибуты и как пишется каждый. Блок класса подмешивается к строке по её классу.
    """
    blocks: dict[str, str] = {}
    for group in groups:
        if group.get("kind") == "class" and group.get("type") != "global":
            name = str(group.get("attribute", "")).strip()
            body = "\n".join(str(r).strip() for r in group.get("rules", []))
            if name and body:
                blocks[name] = (blocks[name] + "\n" + body) if name in blocks else body
    return blocks


def _pages(doc_dir: Path) -> list[dict]:
    """Блоки страниц документа (data_after_check.json) — по одному словарю на страницу."""
    pages = read_json(doc_dir, "data_after_check.json", [])
    return [parse_page_content(p.get("content")) or {} for p in pages if isinstance(p, dict)]


def clause_index(global_rules: str, entries: list[GroupEntry], doc_dir: Path) -> dict[str, str]:
    """Номер пункта → его текст, по всему документу, чтобы дотянуть пункт, на который ссылается правило.

    Источники — строки групп, начинающиеся с номера пункта, и абзацы страниц документа (вместе с
    идущими за пунктом EXAMPLE/NOTE): группировка номера пунктов сохраняет не всегда.
    """
    index: dict[str, str] = {}
    for body in [global_rules] + [e.body for e in entries]:
        for line in body.splitlines():
            match = _CLAUSE_DEF_RE.match(line)
            if match:
                index.setdefault(match.group(1), line.strip())
    from_pages: dict[str, str] = {}
    last = ""
    for page in _pages(doc_dir):
        for block in page.get("blocks", []):
            if block.get("type") != "paragraph":
                continue
            text = str(block.get("text", "")).strip()
            match = _CLAUSE_DEF_RE.match(text)
            if match:
                last = match.group(1)
                from_pages.setdefault(last, text)
            elif last and _ATTACHED_RE.match(text) and last in from_pages:
                from_pages[last] += "\n" + text
    return {**index, **from_pages}


def opening_text(doc_dir: Path) -> str:
    """Начало документа — текст блоков первых страниц по порядку, без таблиц.

    Область действия документа стоит в его начале, но не всегда правилом: у служебной записки это
    строка темы, у стандарта — заголовок и пункт «Scope»; группировка такую шапку в правила не берёт.
    """
    parts: list[str] = []
    size = 0
    for page in _pages(doc_dir):
        for block in page.get("blocks", []):
            text = str(block.get("text") or "").strip()
            if block.get("type") == "table" or not text:
                continue
            parts.append(text)
            size += len(text)
            if size >= OPENING_CHARS:
                return "\n".join(parts)[:OPENING_CHARS]
    return "\n".join(parts)


def load_rule_set(doc_dir: Path, index_source=None) -> RuleSet | None:
    """Документ из его папки извлечения; None, если в нём нет ни одной группы атрибутов.

    index_source — откуда строить поисковые индексы запасного пути (объект с embedding_model и
    vector_store_dir, обычно Settings).
    """
    name = doc_dir.name
    groups: list[dict] = json.loads((doc_dir / "rule_groups.json").read_text(encoding="utf-8"))
    global_rules, entries = split_groups(groups)
    if not entries:
        return None
    tables = load_tables(doc_dir)
    legacy = legacy_fields(doc_dir, groups, name, tables)
    # форма значений группы — по примерам её карточки
    for entry in entries:
        card_examples = (legacy["cards"].get(entry.attribute) or {}).get("examples") or []
        entry.shapes = {sh for e in card_examples for sh in (value_shape(e["from"]), value_shape(e["to"])) if sh}
    parent_to_indices: dict[str, list[int]] = {}
    for i, e in enumerate(entries):
        parent_to_indices.setdefault(e.parent or e.attribute, []).append(i)
    group_subjects = read_json(doc_dir, "subjects.json", {})
    if len({s for s in group_subjects.values() if s}) > 1:
        logger.info("Документ %r описывает несколько предметов: %s", name, sorted(set(group_subjects.values())))
    return RuleSet(name=name, global_rules=global_rules, groups=entries, index_source=index_source,
                   parent_to_indices=parent_to_indices, clause_index=clause_index(global_rules, entries, doc_dir),
                   group_subjects=group_subjects, tables=tables, class_blocks=class_blocks(groups),
                   opening_text=opening_text(doc_dir), **legacy)


def load_all_rule_sets(documents_dir: Path, names: set[str] | None = None, index_source=None) -> list[RuleSet]:
    """Все документы из documents_dir (папки с rule_groups.json); names — только эти документы."""
    rule_sets: list[RuleSet] = []
    for path in sorted(documents_dir.glob("*/rule_groups.json")):
        if names is not None and path.parent.name not in names:
            continue
        rule_set = load_rule_set(path.parent, index_source)
        if rule_set is not None:
            rule_sets.append(rule_set)
            RULE_SET_LIBRARY[document_id(rule_set.name)] = rule_set
    return rule_sets
