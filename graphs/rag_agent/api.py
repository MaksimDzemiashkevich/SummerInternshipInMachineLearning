"""FastAPI-приложение: эндпоинты /chat, /standards, /process. Сам конвейер — в pipeline/."""
import base64
import io
import json
import re
from functools import lru_cache
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .modules.agent import build_tools, run_agent
from .modules.chat import history_messages
from .modules.config import AppConfig, Settings, build_app_config
from .modules.workspace import get_workspace
from .pipeline.common import logger
from .pipeline.excel import XLSX_MEDIA_TYPE, _write_excel_with_highlight
from .pipeline.runner import normalize_table
from .pipeline.user_rules import user_rule_hits

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

app = FastAPI()
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
    output_buffer = _write_excel_with_highlight(result_df, ambiguous_cols)

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
    return {"standards": [{"name": rs.name, "label": _display_name(rs.name)}
                          for rs in app_config.rule_sets]}


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


def _rule_sets_by_name(app_config: AppConfig, names: list[str]) -> list:
    """Имена → наборы правил. Пустой список (и список из одних неизвестных) — значит все."""
    by_name = {rs.name: rs for rs in app_config.rule_sets}
    chosen = [by_name[n] for n in names if n in by_name]
    return chosen or list(app_config.rule_sets)


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
    text = (message or "").strip()
    uploaded = _uploaded(file)

    if not text and uploaded is None:
        return {"reply": "Напишите сообщение или приложите файл .xlsx — обработаю.",
                "session_id": session, "files": []}

    notes: list[str] = []
    if uploaded is not None:
        try:
            frame = pd.read_excel(io.BytesIO(uploaded.file.read()))
        except Exception as e:
            logger.warning("[CHAT] файл %r не прочитан: %s", uploaded.filename, e)
            return {"reply": f"Не удалось прочитать «{uploaded.filename}». Нужен .xlsx с "
                             f"колонками Attribute и Value.",
                    "session_id": session, "files": []}
        stored = workspace.add(name=uploaded.filename, frame=frame)
        columns = ", ".join(str(c) for c in frame.columns)
        notes.append(f"The user has just attached the file {stored.name} "
                     f"(file_id={stored.file_id}), {frame.shape[0]} rows, columns: {columns}.")

    selected = _selected_standard_names(app_config, standards)
    notes.append("Standards picked in the interface: "
                 + (", ".join(selected) if selected else "none, so all of them") + ".")

    produced: list[str] = []
    catalogue = [{"name": rs.name, "label": _display_name(rs.name)} for rs in app_config.rule_sets]
    tools = build_tools(
        workspace, catalogue, selected,
        lambda frame, names, rules: normalize_table(
            frame, app_config, _rule_sets_by_name(app_config, names), rules),
        produced,
        lambda frame, rules: user_rule_hits(frame, rules, app_config.model),
    )
    prompt = f"[Context: {' '.join(notes)}]\n\n{text or 'Обработай приложенный файл.'}"
    reply = run_agent(app_config.assistant or app_config.model, tools,
                      history_messages(_parse_json_list(history)), prompt)

    files = []
    for file_id in produced:
        stored = workspace.get(file_id)
        if stored is None:
            continue
        buffer = _write_excel_with_highlight(stored.frame, stored.ambiguous)
        files.append({
            "name": _download_name(stored),
            "media_type": XLSX_MEDIA_TYPE,
            "content_base64": base64.b64encode(buffer.read()).decode("ascii"),
        })

    return {"reply": reply, "session_id": session, "files": files}
