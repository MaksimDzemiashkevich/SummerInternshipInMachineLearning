"""Правила, названные пользователем в разговоре, и перевод истории чата в сообщения модели.

Такие правила не попадают в базу знаний: агент передаёт их в нормализацию тем запуском,
которым считает нужным, и они ложатся отдельной колонкой рядом с колонками стандартов.
"""

from __future__ import annotations

from dataclasses import dataclass

# Сколько последних реплик уходит в модель: чат ведёт диалог, а не хранит длинную память.
MAX_HISTORY_MESSAGES = 12


@dataclass(frozen=True)
class AdHocRule:
    """Правило пользователя. attribute пуст — правило общее для всех строк."""
    text: str
    attribute: str = ""


def format_adhoc_rules(rules: list[AdHocRule]) -> str:
    """Правила пользователя как текст правил для промпта нормализации."""
    if not rules:
        return ""
    lines = []
    for rule in rules:
        prefix = f"For the attribute {rule.attribute}: " if rule.attribute else ""
        lines.append(f"- {prefix}{rule.text}")
    return ("Rules supplied by the user for this request. They are the only rules that govern "
            "this value: apply them and change nothing else.\n" + "\n".join(lines))


def history_messages(history: list[dict]) -> list[dict]:
    """История чата с фронта → сообщения модели. Роль 'ai' — это assistant."""
    messages = []
    for item in history[-MAX_HISTORY_MESSAGES:]:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", "")).strip()
        if not content:
            continue  # вложение без текста модели нечего сказать
        role = "assistant" if item.get("role") in ("ai", "assistant") else "user"
        messages.append({"role": role, "content": content})
    return messages
