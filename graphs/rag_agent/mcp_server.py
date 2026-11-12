"""MCP-сервер: те же инструменты, что у агента чата, для любого MCP-клиента.

Смонтирован в FastAPI (api.py) по адресу http://localhost:8000/mcp/ (транспорт streamable HTTP):
файлы диалога живут в памяти процесса, поэтому сервер обязан работать в том же процессе, что и
чат. Каждый инструмент принимает session_id — рабочую область, с которой он работает; файл в неё
кладёт upload_excel, забирает download_file (содержимое base64).

Подтверждение записи в базу правил здесь не ждёт «следующего сообщения» — у MCP-клиента их нет:
клиент сам спрашивает пользователя перед вызовом confirm_pending.
"""

import base64
import inspect

from mcp.server.fastmcp import FastMCP

from .modules.toolbox import Toolbox
from .modules.workspace import get_workspace, save_workspace

mcp = FastMCP(
    "normalization",
    instructions=("Tools of a service that standardizes spreadsheet attribute values against "
                  "technical standards. Start with upload_excel (or list_files for a session that "
                  "already has files), then normalize_file, then ask questions with file_overview, "
                  "top_problems, list_issues, explain_row. Never invent normalized values. Changes of "
                  "a standard's rules go propose_rule -> show the preview to the user -> "
                  "confirm_pending only after the user agrees."),
    streamable_http_path="/",
    stateless_http=True,
    json_response=True,
)


def _toolbox(session_id: str) -> Toolbox:
    """Инструменты агента для сессии MCP; подтверждение без ожидания следующей реплики."""
    from .api import catalogue, get_app_config
    app_config = get_app_config()
    return Toolbox(get_workspace(session_id), app_config, catalogue(app_config),
                   require_new_turn=False, session_id=session_id)


def _register(name: str) -> None:
    """Инструмент Toolbox → инструмент MCP с первым аргументом session_id."""
    method = getattr(Toolbox, name)
    signature = inspect.signature(method)
    session = inspect.Parameter("session_id", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=str)
    parameters = [session] + [p for p in signature.parameters.values() if p.name != "self"]

    def tool(session_id: str, **kwargs):
        try:
            return getattr(_toolbox(session_id), name)(**kwargs)
        finally:
            save_workspace(session_id)

    tool.__signature__ = signature.replace(parameters=parameters)
    tool.__name__ = name
    description = (method.__doc__ or name) + (
        "\n\n`session_id`: the workspace to work in — any string you choose; use the same one "
        "for every call of one conversation.")
    mcp.add_tool(tool, name=name, description=inspect.cleandoc(description))


for _name in Toolbox.TOOL_NAMES:
    _register(_name)


@mcp.tool()
def upload_excel(session_id: str, name: str, content_base64: str) -> dict:
    """Put an .xlsx file (base64) into the workspace of this session; returns its file_id.

    A table needs the columns Attribute and Value; other columns are row context. A result made
    by this service earlier is recognized and can be queried and edited.
    """
    from .api import store_upload
    stored = store_upload(get_workspace(session_id), name, base64.b64decode(content_base64))
    save_workspace(session_id)
    return {"file": stored.summary()}


@mcp.tool()
def download_file(session_id: str, file_id: str) -> dict:
    """Return a file of the workspace as .xlsx (base64), with flagged cells filled yellow."""
    from .pipeline.excel import _write_excel_with_highlight
    stored = get_workspace(session_id).get(file_id)
    if stored is None:
        return {"error": f"no file {file_id}"}
    buffer = _write_excel_with_highlight(stored.frame, stored.ambiguous, stored.layout)
    return {"name": stored.name, "content_base64": base64.b64encode(buffer.read()).decode("ascii")}
