from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import json

# Загружаем .env один раз при импорте config
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Статические настройки приложения."""
    base_dir: Path = Path(".")
    input_excel_path: Path = Path("test_attributes.xlsx")
    output_excel_path: Path = Path("test_first_phase_table2.xlsx")
    rules_json_path: Path = Path("rules.json")
    rules_txt_path: Path = Path("rules.txt")
    vector_store_path: Path = Path("vector_store.json")

    chat_model: str = "gpt-4o"
    router_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-large"
    temperature: float = 0.0


@dataclass
class AppConfig:
    """Runtime-зависимости, которые передаются в граф."""
    settings: Settings
    data_frame: pd.DataFrame
    store: InMemoryVectorStore
    model: ChatOpenAI
    router_model: ChatOpenAI


def load_rules(settings: Settings) -> dict[str, str]:

    if settings.rules_json_path.exists():
        with open(settings.rules_json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    with open(settings.rules_txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = text.split("\n\n\n")[2:]
    rules: dict[str, str] = {}

    for chunk in chunks:
        parts = chunk.split("\n\n")
        if len(parts) < 2:
            continue
        header = parts[0].strip()
        body = parts[1].strip()
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

    data_frame = pd.read_excel(settings.input_excel_path)
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
        data_frame=data_frame,
        store=store,
        model=model,
        router_model=router_model,
    )