"""Разговорный слой над нормализацией: свободный чат и разовые правила пользователя.

Сообщение пользователя может быть чем угодно — приветствием, вопросом, сопроводительным
текстом к файлу — и при этом называть собственные правила («у меня длины пишутся в мм»).
Такие правила не попадают в базу знаний: они действуют только на тот запрос, в котором
названы, и подаются модели рядом с правилами документа.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from .system_prompts import CHAT_PROMPT, RULE_EXTRACTION_PROMPT

logger = logging.getLogger("normalization_pipeline")

# Сколько последних реплик уходит в модель: чат ведёт диалог, а не хранит длинную память.
MAX_HISTORY_MESSAGES = 12


@dataclass(frozen=True)
class AdHocRule:
    """Правило, названное в самом сообщении. attribute пуст — правило общее для всех строк."""
    text: str
    attribute: str = ""


def _parse_json_object(raw) -> dict:
    """JSON-объект из ответа модели; markdown-ограждение снимается, мусор даёт пустой dict."""
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", str(raw).strip()).strip()
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def extract_adhoc_rules(model, message: str) -> list[AdHocRule]:
    """Достаёт из сообщения правила пользователя; их отсутствие — нормальный исход."""
    if model is None or not message.strip():
        return []
    messages = [{"role": "system", "content": RULE_EXTRACTION_PROMPT},
                {"role": "user", "content": message}]
    try:
        parsed = _parse_json_object(model.invoke(messages).content)
    except Exception as e:  # сеть/модель: разговор продолжается и без разбора правил
        logger.warning("[CHAT] правила из сообщения не разобраны: %s", e)
        return []

    rules: list[AdHocRule] = []
    for item in parsed.get("rules") or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("rule", "")).strip()
        if text:
            rules.append(AdHocRule(text=text, attribute=str(item.get("attribute", "")).strip()))
    if rules:
        logger.info("[CHAT] правила пользователя на этот запрос: %s",
                    [(r.attribute or "*", r.text) for r in rules])
    return rules


def format_adhoc_rules(rules: list[AdHocRule]) -> str:
    """Блок правил пользователя для промпта нормализации (пусто, если правил нет)."""
    if not rules:
        return ""
    lines = []
    for rule in rules:
        prefix = f"For the attribute {rule.attribute}: " if rule.attribute else ""
        lines.append(f"- {prefix}{rule.text}")
    return ("Additional rules supplied by the user for this request only. They carry the same "
            "force as the rules above and take precedence over them wherever the two conflict:\n"
            + "\n".join(lines))


def _history_messages(history: list[dict]) -> list[dict]:
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


def chat_reply(model, message: str, history: list[dict] | None = None,
               rules: list[AdHocRule] | None = None) -> str:
    """Ответ в свободном разговоре — когда файла в запросе нет и нормализовать нечего."""
    messages = [{"role": "system", "content": CHAT_PROMPT}]
    messages += _history_messages(history or [])
    if rules:
        # Правила живут ровно один запрос, поэтому названные без файла ничего не нормализуют:
        # честнее сказать это сразу, чем молча принять их и потерять.
        listing = "; ".join(r.text for r in rules)
        messages.append({"role": "system", "content": (
            "The user's message states rules of their own: " + listing + ". A rule applies only "
            "to the request it is sent with, and this request has no attached file, so nothing "
            "was normalized. Acknowledge the rules briefly, restate them so the user can check "
            "you read them correctly, and say they must be sent together with the file to take "
            "effect.")})
    messages.append({"role": "user", "content": message})

    try:
        reply = str(model.invoke(messages).content).strip()
    except Exception as e:
        logger.warning("[CHAT] ответ модели не получен: %s", e)
        return "Не удалось получить ответ модели. Попробуйте ещё раз."
    return reply or "…"
