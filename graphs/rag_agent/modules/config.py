from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import json

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
class RuleSet:
    """Правила одного rule-файла: свой векторный индекс + глобальные правила."""
    name: str
    store: InMemoryVectorStore
    global_rules: str


@dataclass
class AppConfig:
    settings: Settings
    model: ChatOpenAI
    rule_sets: list[RuleSet]


# Attribute names that mark document front/back matter rather than technical rules.
# Deliberately generic so it applies to any standards/specification document.
# This is a best-effort safety net; the primary defense against non-normative
# content is the extraction prompt.
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


def filter_non_rule_groups(rules: dict[str, str]) -> dict[str, str]:
    """Drop groups that describe document front/back matter, not technical rules."""
    return {
        header: body
        for header, body in rules.items()
        if not any(marker in header.lower() for marker in _NON_RULE_ATTRIBUTE_MARKERS)
    }


def _split_groups(groups: list[dict]) -> tuple[str, dict[str, str]]:
    """Разбивает группы одного документа на (global_rules_text, local_rules_dict).

    global — правила, действующие на все атрибуты (единицы, нотация, общие принципы).
    local  — правила конкретного атрибута, идут в векторный индекс.
    """
    global_rules: list[str] = []
    local_rules: dict[str, str] = {}

    for group in groups:
        header = group.get("attribute", "").strip()
        body = "\n".join(rule.strip() for rule in group.get("rules", []))

        if group.get("type") == "global":
            global_rules.append(body)
        else:
            if header in local_rules:
                local_rules[header] = local_rules[header] + "\n" + body
            else:
                local_rules[header] = body

    return "\n".join(global_rules), filter_non_rule_groups(local_rules)


def load_vector_store(settings: Settings, rules: dict[str, str], name: str) -> InMemoryVectorStore:
    """Строит (или грузит из кэша) векторный индекс для одного rule-файла."""
    embeddings = OpenAIEmbeddings(model=settings.embedding_model)
    cache_path = settings.vector_store_dir / f"vector_store_{name}.json"

    if cache_path.exists():
        return InMemoryVectorStore.load(str(cache_path), embeddings)

    store = InMemoryVectorStore(embeddings)
    docs = [
        Document(page_content=header, metadata={"text": body})
        for header, body in rules.items()
    ]
    store.store.clear()
    store.add_documents(docs)
    store.dump(str(cache_path))
    return store


def load_all_rule_sets(settings: Settings) -> list[RuleSet]:
    """Находит все rule_groups_*.json и собирает по одному RuleSet на каждый файл."""
    rule_sets: list[RuleSet] = []

    for path in sorted(settings.rule_groups_dir.glob("rule_groups_*.json")):
        name = path.stem[len("rule_groups_"):]
        groups: list[dict] = json.loads(path.read_text(encoding="utf-8"))
        global_rules, local_rules = _split_groups(groups)
        store = load_vector_store(settings, local_rules, name)
        rule_sets.append(RuleSet(name=name, store=store, global_rules=global_rules))

    return rule_sets


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
