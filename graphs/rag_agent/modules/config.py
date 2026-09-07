from __future__ import annotations
from dataclasses import dataclass
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
    """Одна группа правил: имя атрибута, алиасы и текст правил."""
    attribute: str
    aliases: list[str]
    body: str


@dataclass
class RuleSet:
    """Правила одного rule-файла: semantic-индекс + BM25 + список групп."""
    name: str
    global_rules: str
    store: InMemoryVectorStore
    bm25: "BM25Okapi"
    groups: list[GroupEntry]


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
            entries.append(GroupEntry(attribute=header, aliases=aliases, body=body))

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
        store = load_vector_store(settings, entries, name)
        corpus = [_tokenize(_search_text(e)) for e in entries]
        bm25 = BM25Okapi(corpus)
        rule_sets.append(RuleSet(name=name, global_rules=global_rules, store=store, bm25=bm25, groups=entries))

    return rule_sets


def hybrid_search(rule_set: RuleSet, query: str, top_k: int = 4,
                  k_semantic: int = 8, k_bm25: int = 8) -> list[str]:
    """Гибридный поиск: semantic + BM25, слияние через RRF. Возвращает тела правил."""
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

    top = sorted(rrf.items(), key=lambda kv: -kv[1])[:top_k]
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
