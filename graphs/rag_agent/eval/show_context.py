# -*- coding: utf-8 -*-
"""Показывает, что именно получит модель по одной строке таблицы.

Видно всё, что собирается до вызова: какие группы правил нашёл ретрив, какие пункты
подтянуты по ссылкам, какая таблица кодов подложена и пошёл ли детерминированный путь.

Запуск (из корня репозитория):

    venv2/Scripts/python -m graphs.rag_agent.eval.show_context --attr "Pipe schedule" --value "Schedule 40 or 80"
    venv2/Scripts/python -m graphs.rag_agent.eval.show_context --attr "Facing type" --value "Face, raised" \
        --context "Item: WN Flange NPS6 CL600" --doc OGX
    venv2/Scripts/python -m graphs.rag_agent.eval.show_context --attr "Nominal size, DN" --value "DN 150" --full
"""
from __future__ import annotations

import argparse

from ..api import apply_deterministic, build_rules_text
from ..modules.config import build_app_config, Settings

CLAUSES_MARKER = "Clauses referenced above"
REGISTER_MARKER = "Register entries this document defines"


def _section(text: str, marker: str) -> str:
    """Кусок собранного контекста, начинающийся с этого маркера."""
    start = text.find(marker)
    if start < 0:
        return ""
    end = text.find("\n\n", text.find("\n", start))
    return text[start:end if end > 0 else len(text)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attr", required=True, help="подпись атрибута, как в таблице")
    parser.add_argument("--value", required=True, help="значение из ячейки")
    parser.add_argument("--context", default="", help="уточнение строки, например 'Item: gate valve'")
    parser.add_argument("--doc", default="", help="часть имени документа (по умолчанию — все)")
    parser.add_argument("--full", action="store_true", help="печатать весь текст правил целиком")
    args = parser.parse_args()

    app_config = build_app_config(Settings())
    for rule_set in app_config.rule_sets:
        if args.doc and args.doc.lower() not in rule_set.name.lower():
            continue

        print("=" * 100)
        print(f"ДОКУМЕНТ: {rule_set.name}")

        deterministic = apply_deterministic(args.value, rule_set, args.attr, model=app_config.model)
        if deterministic is not None:
            value, needs_review, note = deterministic
            print(f"  детерминированный путь -> {value!r} | на проверку: {needs_review}")
            print(f"  применено: {note}")
            print("  (модель по этой строке не вызывается, контекст правил не собирается)")
            continue

        text, names, conflict = build_rules_text(
            rule_set, args.attr, args.value, args.context, model=app_config.model)
        print(f"  найденные группы правил: {names}")
        print(f"  смешение предметов: {conflict}")
        print(f"  размер контекста: {len(text)} символов")

        clauses = _section(text, CLAUSES_MARKER)
        print(f"\n  ПУНКТЫ ПО ССЫЛКАМ: {'нет' if not clauses else ''}")
        for line in clauses.splitlines()[1:]:
            print(f"    {line[:150]}")

        register = _section(text, REGISTER_MARKER)
        print(f"\n  ТАБЛИЦА КОДОВ: {'нет' if not register else ''}")
        for line in register.splitlines()[1:]:
            print(f"    {line[:150]}")

        if args.full:
            print("\n  --- полный текст правил ---")
            print(text)


if __name__ == "__main__":
    main()
