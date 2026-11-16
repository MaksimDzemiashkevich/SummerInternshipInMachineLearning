"""Поиск групп правил документа по строке таблицы: семантический индекс + BM25, слияние RRF.

Запасной путь: им пользуется только нормализация графом (pipeline/graph.py) для документов без
паспортов атрибутов. Путь по паспортам берёт правила атрибута прямо из его паспорта.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

if TYPE_CHECKING:
    from .rule_set import GroupEntry, RuleSet


def search_text(entry: "GroupEntry") -> str:
    """Текст группы для индексов: имя атрибута и его алиасы."""
    return " | ".join([entry.attribute] + entry.aliases)


def tokenize(text: str) -> list[str]:
    """Слова строки в нижнем регистре — токены для BM25."""
    return re.findall(r"\w+", text.lower())


def load_vector_store(embedding_model: str, cache_dir, entries: list["GroupEntry"], name: str) -> InMemoryVectorStore:
    """Семантический индекс групп документа: из кэша на диске или построенный заново (и сохранённый).

    Кэш — cache_dir/vector_store_<документ>.json; удалите его, чтобы перестроить индекс после
    изменения правил документа.
    """
    embeddings = OpenAIEmbeddings(model=embedding_model)
    cache_path = cache_dir / f"vector_store_{name}.json"
    if cache_path.exists():
        return InMemoryVectorStore.load(str(cache_path), embeddings)
    store = InMemoryVectorStore(embeddings)
    store.add_documents([Document(page_content=search_text(e), metadata={"text": e.body, "index": i})
                         for i, e in enumerate(entries)])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    store.dump(str(cache_path))
    return store


def hybrid_search(rule_set: "RuleSet", query: str, top_k: int = 4,
                  k_semantic: int = 8, k_bm25: int = 8, min_score: float = 0.02) -> list[str]:
    """Тексты top_k групп, ближайших к запросу: ранги семантического поиска и BM25 сливаются RRF.

    Группа со скором RRF ниже min_score отбрасывается: нерелевантные правила в промпте модели
    мешают больше, чем их отсутствие.
    """
    store, bm25 = rule_set.search_index()
    semantic = store.similarity_search(query, k=k_semantic)
    semantic_ranks = [(d.metadata.get("index", -1), rank) for rank, d in enumerate(semantic, start=1)]
    scores = bm25.get_scores(tokenize(query))
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    bm25_ranks = [(i, rank) for rank, i in enumerate(order[:k_bm25], start=1)]

    rrf: dict[int, float] = {}
    for index, rank in semantic_ranks + bm25_ranks:
        if 0 <= index < len(rule_set.groups):
            rrf[index] = rrf.get(index, 0.0) + 1.0 / (60 + rank)
    top = [(i, s) for i, s in sorted(rrf.items(), key=lambda kv: -kv[1])[:top_k] if s >= min_score]
    return [rule_set.groups[i].body for i, _ in top]
