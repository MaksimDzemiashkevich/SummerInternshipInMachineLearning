"""Инструменты над файлами диалога и базой правил — один набор для агента чата и для MCP-сервера.

Каждый метод Toolbox — один инструмент: возвращает словарь (его увидит модель), а его docstring —
описание инструмента для модели, поэтому он по-английски. Агент чата (modules/agent.py) и MCP-сервер
(mcp_server.py) берут их отсюда, так что инструменты не расходятся между двумя входами.

Нормализованных значений здесь никто не придумывает: их считает конвейер, а сводки — код.
Запись в базу правил идёт в два шага: предпросмотр с токеном, затем подтверждение. В чате
подтверждение принимается только в следующем сообщении пользователя — модель не может сама
сказать за него «да» в том же ходе.
"""
from .file_tools import FileTools
from .result_tools import ResultTools
from .rule_tools import RuleTools


class Toolbox(FileTools, ResultTools, RuleTools):
    """Инструменты одного диалога: все наборы вместе. Каждое имя из TOOL_NAMES — инструмент."""

    TOOL_NAMES = ("list_files", "list_standards", "inspect_file", "list_values", "normalize_file",
                  "extract_document", "list_jobs",
                  "file_overview", "top_problems", "list_issues", "explain_row",
                  "renormalize_rows", "edit_result", "accept_proposals",
                  "list_rules", "propose_rule", "propose_rule_removal",
                  "confirm_pending", "cancel_pending")
