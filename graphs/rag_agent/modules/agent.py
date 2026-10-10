"""Агент чата: цикл вызовов модели над инструментами диалога.

Нормализацию агент не делает сам — он решает, ЧТО и по каким стандартам считать, и зовёт
пайплайн инструментом. Всё, что он говорит пользователю про его файлы, приходит из ответа
инструмента: выдумывать нормализованные значения ему запрещено промптом и нечем. Сами
инструменты — в modules/toolbox.py (тот же набор отдаёт MCP-сервер).
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import ToolMessage
from langchain_core.tools import StructuredTool

from .system_prompts import AGENT_PROMPT
from .toolbox import Toolbox

logger = logging.getLogger("normalization_pipeline")

# Потолок шагов цикла: цепочка «показать → правило → предпросмотр → пересчёт → отчёт» требует
# десятка шагов, но крутиться вечно агент не должен.
MAX_AGENT_STEPS = 16


def _dump(payload) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def build_tools(toolbox: Toolbox) -> list:
    """Методы Toolbox → инструменты LangChain (описание — docstring метода)."""
    return [_tool(getattr(toolbox, name), name) for name in Toolbox.TOOL_NAMES]


def _tool(method, name: str) -> StructuredTool:
    """Один инструмент: схема аргументов — из сигнатуры метода, ответ — JSON строкой."""
    schema_source = StructuredTool.from_function(func=method, name=name,
                                                 description=method.__doc__ or name)

    def call(**kwargs):
        return _dump(method(**kwargs))

    return StructuredTool(name=name, description=schema_source.description,
                          args_schema=schema_source.args_schema, func=call)


def run_agent(model, tools: list, history: list[dict], message: str,
              max_steps: int = MAX_AGENT_STEPS) -> str:
    """Цикл «модель ↔ инструменты»; возвращает финальный текст для пользователя."""
    by_name = {t.name: t for t in tools}
    bound = model.bind_tools(tools)
    messages: list = [{"role": "system", "content": AGENT_PROMPT}]
    messages += history
    messages.append({"role": "user", "content": message})

    last_text = ""
    for step in range(max_steps):
        try:
            response = bound.invoke(messages)
        except Exception as e:
            logger.warning("[AGENT] шаг %d не удался: %s", step + 1, e)
            return last_text or "Не удалось получить ответ модели. Попробуйте ещё раз."

        last_text = str(response.content or "").strip() or last_text
        calls = getattr(response, "tool_calls", None)
        if not calls:
            return last_text or "…"

        messages.append(response)
        for call in calls:
            tool_fn = by_name.get(call["name"])
            if tool_fn is None:
                content = _dump({"error": f"unknown tool {call['name']}"})
            else:
                try:
                    content = str(tool_fn.invoke(call["args"]))
                except Exception as e:
                    logger.warning("[AGENT] %s(%s) упал: %s", call["name"], call["args"], e)
                    content = _dump({"error": str(e)})
            logger.info("[AGENT] %s(%s) -> %s", call["name"], call["args"], content[:200])
            messages.append(ToolMessage(content=content, tool_call_id=call["id"]))

    logger.warning("[AGENT] исчерпан лимит шагов (%d)", max_steps)
    return last_text or ("Не уложился в отведённые шаги. Скажите точнее, что нужно сделать: "
                         "какой файл и какие стандарты.")
