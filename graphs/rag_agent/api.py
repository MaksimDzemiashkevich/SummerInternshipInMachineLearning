"""FastAPI-приложение: эндпоинты /chat, /standards, /process. Сам конвейер — в pipeline/."""
import base64
import io
import json
import re
from contextlib import asynccontextmanager
from functools import lru_cache
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .modules.agent import build_tools, run_agent
from .modules.chat import history_messages
from .modules.config import AppConfig, Settings, build_app_config
from .modules.results import layout_of
from .modules.toolbox import Toolbox
from .modules.workspace import focus_note, get_workspace, save_workspace
from .pipeline.common import logger
from .pipeline.excel import XLSX_MEDIA_TYPE, _write_excel_with_highlight, read_flagged_cells
from .pipeline.runner import normalize_table

# Имена, которые берут из api снаружи (eval-скрипты, show_context, run_eval), остаются
# доступными и отсюда, хотя живут в pipeline/.
from .pipeline.deterministic import apply_deterministic, match_closed_register  # noqa: F401
from .pipeline.frames import prepare_frame, _row_context  # noqa: F401
from .pipeline.graph import graph, run_single_rule_set  # noqa: F401
from .pipeline.guards import _rejection_reason  # noqa: F401
from .pipeline.normalizer import SELF_CONSISTENCY  # noqa: F401
from .pipeline.postprocess import _canonicalize_formatting, _canonicalize_units  # noqa: F401
from .pipeline.retrieval import build_rules_text, expand_references  # noqa: F401
from .pipeline.subjects import _document_covers_row, _row_class  # noqa: F401
from .pipeline.user_rules import APPLIED_SUFFIX, USER_RULES_COLUMN, run_user_rules_column  # noqa: F401

@asynccontextmanager
async def _lifespan(_app):
    # сессии MCP-эндпоинта: у смонтированного приложения своя lifespan не запускается
    async with mcp.session_manager.run():
        yield


app = FastAPI(lifespan=_lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache
def get_app_config() -> AppConfig:
    return build_app_config(Settings())


@app.post("/process")
def process_excel(
        file: UploadFile = File(...),
        app_config: AppConfig = Depends(get_app_config),
):
    contents = file.file.read()
    data_frame = pd.read_excel(io.BytesIO(contents))
    result_df, ambiguous_cols = normalize_table(data_frame, app_config)
    output_buffer = _write_excel_with_highlight(result_df, ambiguous_cols, result_df.attrs.get("layout"))

    return Response(
        content=output_buffer.read(),
        media_type=XLSX_MEDIA_TYPE,
        headers={"Content-Disposition": "attachment; filename=result.xlsx"},
    )


_LABEL_TAIL_RE = re.compile(r"\s*(?:normalization\s+)?rules$", re.I)


def _display_name(name: str) -> str:
    """Имя папки документа → ярлык для интерфейса: подчёркивания в пробелы, хвост долой."""
    label = _LABEL_TAIL_RE.sub("", name.replace("_", " ")).strip()
    return re.sub(r"\s+", " ", label) or name


@app.get("/standards")
def list_standards(app_config: AppConfig = Depends(get_app_config)):
    """Какие стандарты загружены — фронт даёт из них выбрать."""
    return {"standards": catalogue(app_config)}


def _parse_json_list(raw: str) -> list:
    """Список приходит с фронта строкой JSON; его порча не должна ронять запрос."""
    try:
        parsed = json.loads(raw) if raw else []
    except ValueError:
        return []
    return parsed if isinstance(parsed, list) else []


def _selected_standard_names(app_config: AppConfig, raw: str) -> list[str]:
    """Имена стандартов, выбранных в интерфейсе; неизвестные отбрасываем."""
    known = {rs.name for rs in app_config.rule_sets}
    return [n for n in (str(x).strip() for x in _parse_json_list(raw)) if n in known]


def catalogue(app_config: AppConfig) -> list[dict]:
    """Загруженные стандарты: имя для инструментов и ярлык для людей."""
    return [{"name": rs.name, "label": _display_name(rs.name)} for rs in app_config.rule_sets]


def store_upload(workspace, name: str, content: bytes):
    """Присланная таблица → файл диалога. Прошлый результат, присланный обратно, узнаётся по
    колонкам «— применено» и жёлтой заливке: по нему можно задавать вопросы и править его."""
    frame = pd.read_excel(io.BytesIO(content))
    layout = layout_of(frame)
    if not layout["standards"]:
        return workspace.add(name=name, frame=frame)
    columns = [e["column"] for e in layout["standards"]]
    if layout["user_rules"]:
        columns.append(layout["user_rules"]["column"])
    return workspace.add(name=name, frame=frame, kind="result", layout=layout,
                         ambiguous=read_flagged_cells(content, columns),
                         origin="результат, присланный пользователем")


def _uploaded(file: UploadFile | None) -> UploadFile | None:
    """Пустое поле формы приходит как файл без имени — это не вложение."""
    return file if file is not None and (file.filename or "").strip() else None


def _download_name(stored) -> str:
    """Имя файла для скачивания: то же, что у файла в рабочей области."""
    name = stored.name.strip() or f"{stored.file_id}.xlsx"
    return name if name.lower().endswith(".xlsx") else f"{name}.xlsx"


@app.post("/chat")
def chat(
        message: str = Form(""),
        history: str = Form("[]"),
        standards: str = Form("[]"),
        session_id: str = Form(""),
        file: UploadFile | None = File(None),
        app_config: AppConfig = Depends(get_app_config),
):
    """Вход чата: сообщение ведёт агент, файлы диалога лежат в его рабочей области.

    Сам агент ничего не нормализует — он выбирает файл, стандарты и правила и зовёт
    пайплайн инструментом. Поэтому «пересчитай тот файл, я забыл правило» работает без
    повторной загрузки: файл никуда не девался, пока жив диалог.
    """
    session = (session_id or "").strip() or uuid4().hex
    workspace = get_workspace(session)
    # новое сообщение пользователя: подготовленное в прошлом ходе теперь можно подтвердить
    workspace.turn += 1
    text = (message or "").strip()
    uploaded = _uploaded(file)

    if not text and uploaded is None:
        return {"reply": "Напишите сообщение или приложите файл .xlsx — обработаю.",
                "session_id": session, "files": []}

    notes: list[str] = []
    if uploaded is not None:
        try:
            stored = store_upload(workspace, uploaded.filename, uploaded.file.read())
        except Exception as e:
            logger.warning("[CHAT] файл %r не прочитан: %s", uploaded.filename, e)
            return {"reply": f"Не удалось прочитать «{uploaded.filename}». Нужен .xlsx с "
                             f"колонками Attribute и Value.",
                    "session_id": session, "files": []}
        columns = ", ".join(str(c) for c in stored.frame.columns)
        kind = (" It is a normalization result made earlier (standards: "
                + ", ".join(e["standard"] for e in stored.layout["standards"]) + ")."
                if stored.kind == "result" else "")
        notes.append(f"The user has just attached the file {stored.name} "
                     f"(file_id={stored.file_id}), {stored.frame.shape[0]} rows, columns: {columns}.{kind}")

    selected = _selected_standard_names(app_config, standards)
    notes.append("Standards picked in the interface: "
                 + (", ".join(selected) if selected else "none, so all of them") + ".")

    produced: list[str] = []
    toolbox = Toolbox(workspace, app_config, catalogue(app_config), selected, produced)
    for note in (focus_note(workspace), toolbox.pending_note()):
        if note:
            notes.append(note)
    tools = build_tools(toolbox)
    prompt = f"[Context: {' '.join(notes)}]\n\n{text or 'Обработай приложенный файл.'}"
    try:
        reply = run_agent(app_config.assistant or app_config.model, tools,
                          history_messages(_parse_json_list(history)), prompt)
    finally:
        save_workspace(session)

    files = []
    for file_id in produced:
        stored = workspace.get(file_id)
        if stored is None:
            continue
        buffer = _write_excel_with_highlight(stored.frame, stored.ambiguous, stored.layout)
        files.append({
            "name": _download_name(stored),
            "media_type": XLSX_MEDIA_TYPE,
            "content_base64": base64.b64encode(buffer.read()).decode("ascii"),
        })

    return {"reply": reply, "session_id": session, "files": files}


# MCP: те же инструменты, что у агента чата, для внешних клиентов (см. mcp_server.py)
from .mcp_server import mcp  # noqa: E402

app.mount("/mcp", mcp.streamable_http_app())
