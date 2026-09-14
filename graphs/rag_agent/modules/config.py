from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

import json
import re

from rank_bm25 import BM25Okapi

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .tools import array_of_tools

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Статические настройки приложения."""
    base_dir: Path = BASE_DIR
    rule_groups_dir: Path = BASE_DIR.parent / "structured_documents" / "data" / "output"
    vector_store_dir: Path = BASE_DIR / "data" / "input"

    chat_model: str = "gpt-5.6-luna"
    embedding_model: str = "text-embedding-3-large"
    temperature: float = 0.0
    # Для reasoning-моделей (gpt-5.6-luna) тулзы работают только при reasoning_effort="none".
    reasoning_effort: str | None = "none"


@dataclass
class GroupEntry:
    """Одна группа правил: имя атрибута, алиасы, текст правил и родительская тема."""
    attribute: str
    aliases: list[str]
    body: str
    parent: str = ""


@dataclass
class RuleSet:
    """Правила одного rule-файла: semantic-индекс + BM25 + список групп."""
    name: str
    global_rules: str
    store: InMemoryVectorStore
    bm25: "BM25Okapi"
    groups: list[GroupEntry]
    deterministic: dict[str, str] = field(default_factory=dict)
    phrase_map: dict[str, str] = field(default_factory=dict)
    closed_registers: dict[str, dict[str, str]] = field(default_factory=dict)
    attribute_to_register: dict[str, str] = field(default_factory=dict)
    number_first: bool = False
    deterministic_order: dict[str, bool] = field(default_factory=dict)
    deterministic_attributes: dict[str, str] = field(default_factory=dict)
    excluded_attributes: list[str] = field(default_factory=list)
    render_templates: dict[str, dict] = field(default_factory=dict)
    parent_to_indices: dict[str, list[int]] = field(default_factory=dict)


@dataclass
class AppConfig:
    settings: Settings
    model: ChatOpenAI
    rule_sets: list[RuleSet]


# Attribute names that mark document front/back matter rather than technical rules.
_NON_RULE_ATTRIBUTE_MARKERS = (
    "metadata",
    "copyright",
    "legal",
    "liability",
    "disclaimer",
    "table of contents",
    "bibliography",
    "foreword",
    "acknowledgment",
    "contact information",
    "distribution",
    "publisher",
    "provided by",
)


def _is_non_rule(header: str) -> bool:
    return any(marker in header.lower() for marker in _NON_RULE_ATTRIBUTE_MARKERS)


def _split_groups(groups: list[dict]) -> tuple[str, list[GroupEntry]]:
    """Разбивает группы одного документа на (global_rules_text, list[GroupEntry]).

    global — правила, действующие на все атрибуты.
    GroupEntry — local-группы с именем, алиасами и текстом правил (поиск по атрибуту).
    """
    global_rules: list[str] = []
    entries: list[GroupEntry] = []

    for group in groups:
        header = group.get("attribute", "").strip()
        body = "\n".join(rule.strip() for rule in group.get("rules", []))

        if group.get("type") == "global":
            global_rules.append(body)
        else:
            if _is_non_rule(header):
                continue
            aliases = [a.strip() for a in group.get("aliases", []) if a and a.strip()]
            entries.append(GroupEntry(attribute=header, aliases=aliases, body=body, parent=group.get("parent", "") or header))

    return "\n".join(global_rules), entries


def _search_text(entry: GroupEntry) -> str:
    """Текст, который эмбедится и по которому строится BM25: имя + алиасы."""
    return " | ".join([entry.attribute] + entry.aliases)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def load_vector_store(settings: Settings, entries: list[GroupEntry], name: str) -> InMemoryVectorStore:
    """Строит (или грузит из кэша) semantic-индекс: эмбедится имя + алиасы."""
    embeddings = OpenAIEmbeddings(model=settings.embedding_model)
    cache_path = settings.vector_store_dir / f"vector_store_{name}.json"

    if cache_path.exists():
        return InMemoryVectorStore.load(str(cache_path), embeddings)

    store = InMemoryVectorStore(embeddings)
    docs = [
        Document(page_content=_search_text(e), metadata={"text": e.body, "index": i})
        for i, e in enumerate(entries)
    ]
    store.store.clear()
    store.add_documents(docs)
    store.dump(str(cache_path))
    return store


def load_all_rule_sets(settings: Settings) -> list[RuleSet]:
    """Находит все <doc_dir>/rule_groups.json и собирает RuleSet на каждый документ."""
    rule_sets: list[RuleSet] = []

    for path in sorted(settings.rule_groups_dir.glob("*/rule_groups.json")):
        name = path.parent.name
        groups: list[dict] = json.loads(path.read_text(encoding="utf-8"))
        global_rules, entries = _split_groups(groups)
        if not entries:
            continue
        parent_to_indices: dict[str, list[int]] = {}
        for i, e in enumerate(entries):
            parent_to_indices.setdefault(e.parent or e.attribute, []).append(i)
        store = load_vector_store(settings, entries, name)
        corpus = [_tokenize(_search_text(e)) for e in entries]
        bm25 = BM25Okapi(corpus)
        deterministic = _load_deterministic(path.parent)
        phrase_map = _load_phrase_rules(path.parent)
        closed_registers = _build_closed_registers(phrase_map, deterministic)
        attribute_to_register = _derive_attribute_to_register(groups, closed_registers)
        number_first = _determine_number_first(global_rules)
        deterministic_order = _load_deterministic_order(path.parent)
        deterministic_attributes = _load_deterministic_attributes(path.parent)
        excluded_attributes = _load_excluded_attributes(path.parent)
        render_templates = _load_render_templates(path.parent)
        rule_sets.append(RuleSet(name=name, global_rules=global_rules, store=store, bm25=bm25, groups=entries, deterministic=deterministic, phrase_map=phrase_map, closed_registers=closed_registers, attribute_to_register=attribute_to_register, number_first=number_first, deterministic_order=deterministic_order, deterministic_attributes=deterministic_attributes, excluded_attributes=excluded_attributes, render_templates=render_templates, parent_to_indices=parent_to_indices))

    return rule_sets


def _load_deterministic(doc_dir: Path) -> dict[str, str]:
    """Загружает детерминированные правила Keyword → Root из папки документа."""
    det_path = doc_dir / "deterministic_rules.json"
    if det_path.exists():
        return json.loads(det_path.read_text(encoding="utf-8"))
    return {}


def _load_phrase_rules(doc_dir: Path) -> dict[str, str]:
    """Загружает фразовые соответствия «фраза → код» (закрытые списки) из папки документа."""
    phrase_path = doc_dir / "phrase_rules.json"
    if phrase_path.exists():
        return json.loads(phrase_path.read_text(encoding="utf-8"))
    return {}


def _load_excluded_attributes(doc_dir: Path) -> list[str]:
    """Загружает атрибуты, исключённые из реформатирования (reproduced exactly / excluded)."""
    excl_path = doc_dir / "excluded_attributes.json"
    if excl_path.exists():
        return json.loads(excl_path.read_text(encoding="utf-8"))
    return []


def _load_deterministic_order(doc_dir: Path) -> dict[str, bool]:
    """Загружает порядок конкатенации keyword→root (per-keyword), если файл есть."""
    order_path = doc_dir / "deterministic_order.json"
    if order_path.exists():
        return json.loads(order_path.read_text(encoding="utf-8"))
    return {}


def _load_deterministic_attributes(doc_dir: Path) -> dict[str, str]:
    """Загружает атрибут-владельца для каждого keyword (keyword → attribute)."""
    attrs_path = doc_dir / "deterministic_attributes.json"
    if attrs_path.exists():
        return json.loads(attrs_path.read_text(encoding="utf-8"))
    return {}


def _load_render_templates(doc_dir: Path) -> dict[str, dict]:
    """Загружает шаблоны рендера (decimals + suffix) для атрибутов с жёстким форматом."""
    render_path = doc_dir / "render_templates.json"
    if render_path.exists():
        return json.loads(render_path.read_text(encoding="utf-8"))
    return {}


_CODE_PREFIX_RE = re.compile(r"^([A-Z]+)[\-]?\d")

def _build_closed_registers(phrase_map: dict, deterministic: dict) -> dict[str, dict[str, str]]:
    """Группирует фразы закрытых регистров по префиксу кода (ECC/JT/GT/...)."""
    registers: dict[str, dict[str, str]] = {}
    for source in (phrase_map, deterministic):
        for phrase, code in source.items():
            m = _CODE_PREFIX_RE.match(code)
            if not m:
                continue  # не похоже на код таблицы (TRIMMEUS, CLASSATUM и т.п.)
            prefix = m.group(1)
            registers.setdefault(prefix, {})[phrase] = code
    return registers


def _derive_attribute_to_register(groups: list[dict], closed_registers: dict[str, dict[str, str]]) -> dict[str, str]:
    """Выводит attribute (lowercase) -> префикс регистра автономно: ищет в тексте правил группы
    коды вида ECC-101 / JT-02 / GT-05 (только префиксы, реально присутствующие в closed_registers).
    Без хардкода списка атрибутов."""
    prefixes = list(closed_registers.keys())
    if not prefixes:
        return {}
    alt = "|".join(re.escape(p) for p in sorted(prefixes, key=len, reverse=True))
    # маппинг в форме «Code: JT-01», «→ "ECC-101"», «Assigned Codes: ECC-101 …»,
    # а не любое упоминание кода в прозе.
    pat = re.compile(
        r"(?:Code(?:\s+Block)?\s*:\s*|Assigned\s+Codes\s*:\s*|→\s*)"
        r"[“”‘’'\"]?"
        r"(" + alt + r")-?\d+\b"
    )
    mapping: dict[str, str] = {}
    for g in groups:
        if g.get("type") == "global":
            continue
        for r in g.get("rules", []):
            m = pat.search(r)
            if m:
                mapping[g["attribute"].strip().lower()] = m.group(1)
                break
    return mapping


def _determine_number_first(global_rules: str) -> bool:
    """Определяет порядок keyword→root по тексту глобальных правил:
    number-first («numeral first» → 300CLASSATUM) или root-first (CL300)."""
    return bool(re.search(r"numeral\s+first", global_rules.lower()))


def hybrid_search(rule_set: RuleSet, query: str, top_k: int = 4,
                  k_semantic: int = 8, k_bm25: int = 8,
                  min_score: float = 0.02) -> list[str]:
    """Гибридный поиск: semantic + BM25, слияние через RRF. Возвращает тела правил.

    min_score — порог по RRF-score: группы со скором ниже отбрасываются, чтобы не
    тянуть нерелевантный/противоречащий контекст в промпт модели.
    """
    # semantic
    sem_docs = rule_set.store.similarity_search(query, k=k_semantic)
    sem_ranks = [(d.metadata.get("index", -1), rank) for rank, d in enumerate(sem_docs, start=1)]

    # bm25
    scores = rule_set.bm25.get_scores(_tokenize(query))
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    bm25_ranks = [(i, rank) for rank, i in enumerate(order[:k_bm25], start=1)]

    # RRF
    rrf: dict[int, float] = {}
    for idx, rank in sem_ranks + bm25_ranks:
        if 0 <= idx < len(rule_set.groups):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (60 + rank)

    top = [(i, s) for i, s in sorted(rrf.items(), key=lambda kv: -kv[1])[:top_k] if s >= min_score]
    return [rule_set.groups[i].body for i, _ in top]


def build_app_config(settings: Settings | None = None) -> AppConfig:
    settings = settings or Settings()

    model = ChatOpenAI(
        model=settings.chat_model,
        temperature=settings.temperature,
        reasoning_effort=settings.reasoning_effort,
    ).bind_tools(array_of_tools)

    rule_sets = load_all_rule_sets(settings)

    return AppConfig(
        settings=settings,
        model=model,
        rule_sets=rule_sets,
    )
