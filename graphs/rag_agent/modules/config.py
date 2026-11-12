"""Настройки сервиса и его сборка: какие модели создать и какие документы загрузить.

Это единственное место, где создаются модели и загружаются стандарты; остальные модули получают
их готовыми (AppConfig), а не создают сами.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from graphs.paths import DOCUMENTS_OUTPUT, SERVICE_INPUT

from .rule_set import RuleSet, load_all_rule_sets
from .tools import array_of_tools

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Настройки сервиса: где лежат документы и индексы, какие модели и с каким рассуждением."""
    rule_groups_dir: Path = DOCUMENTS_OUTPUT
    vector_store_dir: Path = SERVICE_INPUT

    chat_model: str = "gpt-5.6-luna"
    embedding_model: str = "text-embedding-3-large"
    temperature: float = 0.0
    # модели с привязанными инструментами: через Chat Completions инструменты работают только без рассуждения
    reasoning_effort: str | None = "none"
    # паспорта атрибутов и сборка правил пользователя: инструментов нет, рассуждение повышает точность
    spec_reasoning_effort: str | None = "low"
    # агент в чате: инструменты с рассуждением — через Responses API; "none" — без рассуждения
    agent_reasoning_effort: str | None = os.getenv("AGENT_REASONING_EFFORT", "low")


@dataclass
class AppConfig:
    """Собранный сервис: настройки, модели и загруженные стандарты."""
    settings: Settings
    model: ChatOpenAI            # модель запасного пути: с инструментами пересчёта единиц
    rule_sets: list[RuleSet]
    assistant: ChatOpenAI | None = None   # та же модель без инструментов
    spec_model: ChatOpenAI | None = None  # паспорта атрибутов и сборка правил пользователя
    agent_model: ChatOpenAI | None = None  # агент в чате; None — assistant


def build_app_config(settings: Settings | None = None) -> AppConfig:
    """Сервис по настройкам: модели и все стандарты из папки извлечённых документов."""
    settings = settings or Settings()
    # без таймаута SDK ждёт зависший запрос по 600 с на каждую попытку
    assistant = ChatOpenAI(model=settings.chat_model, temperature=settings.temperature,
                           reasoning_effort=settings.reasoning_effort, timeout=120, max_retries=2)
    spec_model = ChatOpenAI(model=settings.chat_model, temperature=settings.temperature,
                            reasoning_effort=settings.spec_reasoning_effort, timeout=180, max_retries=2)
    agent_model = (ChatOpenAI(model=settings.chat_model, use_responses_api=True,
                              reasoning={"effort": settings.agent_reasoning_effort}, timeout=180, max_retries=2)
                   if settings.agent_reasoning_effort not in (None, "", "none") else None)
    return AppConfig(settings=settings, model=assistant.bind_tools(array_of_tools),
                     rule_sets=load_all_rule_sets(settings.rule_groups_dir, index_source=settings),
                     assistant=assistant, spec_model=spec_model, agent_model=agent_model)


def add_rule_set(app_config: AppConfig, name: str) -> RuleSet | None:
    """Подгружает документ в работающий сервис (после фонового извлечения); заменяет прежний с тем же именем."""
    loaded = load_all_rule_sets(app_config.settings.rule_groups_dir, {name}, index_source=app_config.settings)
    if not loaded:
        return None
    app_config.rule_sets[:] = [rs for rs in app_config.rule_sets if rs.name != name] + loaded
    return loaded[0]
