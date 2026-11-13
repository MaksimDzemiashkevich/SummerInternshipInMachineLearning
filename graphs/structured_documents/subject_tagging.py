# -*- coding: utf-8 -*-
"""Разметка предмета правил: к какому изделию/узлу относится каждая группа.

Один документ может описывать несколько разных предметов, и тогда правила с одинаковым
названием атрибута означают разное. Без разметки применяющий граф видит обе группы сразу и
выбирает между ними наугад. Результат кладётся рядом с rule_groups.json в subjects.json:
{"<имя группы>": "<предмет>"}.

Запуск (из корня репозитория) — не требует перегонки PDF, работает по готовым rule_groups.json:

    venv2/Scripts/python -m graphs.structured_documents.subject_tagging
    venv2/Scripts/python -m graphs.structured_documents.subject_tagging --doc XYZ_STD_205_alternate_normalization_rules
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .modules.config_structured_documents_graph import Settings
from graphs.paths import DOCUMENTS_OUTPUT as OUTPUT_DIR

load_dotenv()

settings = Settings()
# Сколько символов тела группы показывать модели. По 220 символам модель видела лишь начало
# правила и первый пример, и пример про фильтр делал «Airflow rate» правилом фильтров.
SNIPPET = 1200


def _payload(rule_groups: list[dict]) -> str:
    """Общие положения документа + названия локальных групп с фрагментом текста."""
    global_text = "\n".join(
        rule for group in rule_groups if group.get("type") == "global"
        for rule in group.get("rules", [])
    )[:2000]
    groups = [
        {"id": i, "attribute": group.get("attribute", ""),
         "rules": "\n".join(group.get("rules", []))[:SNIPPET]}
        for i, group in enumerate(g for g in rule_groups if g.get("type") != "global")
    ]
    return json.dumps({"document": global_text, "groups": groups}, ensure_ascii=False)


def tag_subjects(rule_groups: list[dict], llm: ChatOpenAI) -> dict[str, str]:
    """Возвращает {имя группы: предмет}. При ошибке — пустой словарь (шаг необязателен)."""
    local = [g for g in rule_groups if g.get("type") != "global"]
    if not local:
        return {}
    try:
        response = llm.invoke([
            SystemMessage(content=settings.SUBJECT_PROMPT),
            HumanMessage(content=_payload(rule_groups)),
        ])
        parsed = json.loads(str(response.content))
    except Exception as e:
        print("SUBJECT TAGGING ERROR (пропускаем документ):", type(e).__name__, str(e)[:200])
        return {}

    # Привязка по номеру группы, а не по названию: названия групп могут повторяться
    # (именно в таком документе разметка и нужна) и модель их не всегда копирует дословно.
    by_index: dict[int, str] = {}
    for subject in parsed.get("subjects", []):
        name = str(subject.get("name", "")).strip().lower()
        if not name:
            continue
        for index in subject.get("groups", []):
            if isinstance(index, int) and 0 <= index < len(local):
                by_index[index] = name
        for attribute in subject.get("attributes", []):  # совместимость со старым форматом
            for i, group in enumerate(local):
                if group.get("attribute", "").strip() == str(attribute).strip():
                    by_index.setdefault(i, name)
    return {group["attribute"]: by_index.get(i, "") for i, group in enumerate(local)}


def main() -> None:
    """CLI: размечает subjects.json для документа --doc или для всех."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", default="", help="имя папки документа (по умолчанию — все)")
    args = parser.parse_args()

    llm = ChatOpenAI(model=settings.llmodel, max_tokens=10000, temperature=0)

    for path in sorted(OUTPUT_DIR.glob("*/rule_groups.json")):
        if args.doc and path.parent.name != args.doc:
            continue
        rule_groups = json.loads(path.read_text(encoding="utf-8"))
        subjects = tag_subjects(rule_groups, llm)
        if not subjects:
            continue
        (path.parent / "subjects.json").write_text(
            json.dumps(subjects, ensure_ascii=False, indent=1), encoding="utf-8")
        distinct = sorted({s for s in subjects.values() if s})
        print(f"{path.parent.name}: групп {len(subjects)}, предметов {len(distinct)} -> {distinct}")


if __name__ == "__main__":
    main()
