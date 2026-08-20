from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import json

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Статические настройки приложения."""
    base_dir: Path = BASE_DIR
    rules_json_path: Path = BASE_DIR / "data" / "input" / "rules.json"
    # rules_txt_path: Path = BASE_DIR / "data" / "input" / "rules.txt"
    vector_store_path: Path = BASE_DIR / "data" / "input" / "vector_store.json"
    rule_groups_path: Path = BASE_DIR.parent / "structured_documents" / "data" / "output" / "rule_groups.json"

    chat_model: str = "gpt-4o"
    router_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-large"
    temperature: float = 0.0


@dataclass
class AppConfig:
    settings: Settings
    store: InMemoryVectorStore
    model: ChatOpenAI
    router_model: ChatOpenAI
    data_frame: pd.DataFrame | None = None


def load_rules(settings: Settings) -> dict[str, str]:
    if settings.rules_json_path.exists():
        with open(settings.rules_json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    with open(settings.rule_groups_path, "r", encoding="utf-8") as f:
        groups: list[dict] = json.load(f)

    rules: dict[str, str] = {}
    for group in groups:
        header = group["attribute"].strip()
        body = "\n".join(rule.strip() for rule in group.get("rules", []))

        if header in rules:
            rules[header] = rules[header] + "\n" + body
        else:
            rules[header] = body

    with open(settings.rules_json_path, "w", encoding="utf-8") as f:
        json.dump(rules, f, ensure_ascii=False, indent=2)

    return rules


def load_vector_store(settings: Settings, rules: dict[str, str]) -> InMemoryVectorStore:
    embeddings = OpenAIEmbeddings(model=settings.embedding_model)

    if settings.vector_store_path.exists():
        return InMemoryVectorStore.load(str(settings.vector_store_path), embeddings)

    store = InMemoryVectorStore(embeddings)
    docs = [
        Document(page_content=header, metadata={"source": "rules.json", "text": body})
        for header, body in rules.items()
    ]
    store.store.clear()
    store.add_documents(docs)
    store.dump(str(settings.vector_store_path))
    return store


def build_app_config(settings: Settings | None = None) -> AppConfig:
    settings = settings or Settings()

    rules = load_rules(settings)
    store = load_vector_store(settings, rules)

    model = ChatOpenAI(
        model=settings.chat_model,
        temperature=settings.temperature,
    )
    router_model = ChatOpenAI(
        model=settings.router_model,
        temperature=settings.temperature,
    )

    return AppConfig(
        settings=settings,
        store=store,
        model=model,
        router_model=router_model,
    )