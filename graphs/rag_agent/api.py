"""FastAPI-приложение: эндпоинты /chat, /standards, /process. Сам конвейер — в pipeline/."""
import base64
import io
import json
import re
from contextlib import asynccontextmanager
from functools import lru_cache
from uuid import uuid4

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .modules import jobs
from .modules.agent import build_tools, run_agent
from .modules.chat import history_messages
from .modules.config import AppConfig, Settings, add_rule_set, build_app_config
from .modules.results import layout_of
from graphs.paths import DOCUMENTS_INPUT as INPUT_DIR

from .modules.toolbox import Toolbox
from .modules.workspace import focus_note, get_workspace, save_workspace
from .pipeline.common import logger
from .pipeline.excel import XLSX_MEDIA_TYPE, _write_excel_with_highlight, read_flagged_cells
from .pipeline.runner import normalize_table

# Имена, которые берут из api снаружи (eval-скрипты, run_eval), остаются
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
    """Жизненный цикл приложения: наблюдатель фоновых задач и сессии MCP-эндпоинта."""
    # наблюдатель за фоновыми извлечениями: очередь, состояние, подгрузка готового документа
    jobs.start_watcher()
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
    """Сервис собирается один раз на процесс: модели и все стандарты."""
    return build_app_config(Settings())


def _load_extracted(job) -> dict:
    """Извлечённый документ — в работающий сервис, без перезапуска; сведения для сообщения в чат."""
    import json as _json
    app_config = get_app_config()
    folder = app_config.settings.rule_groups_dir / job.document
    groups = _json.loads((folder / "rule_groups.json").read_text(encoding="utf-8"))
    specs = (_json.loads((folder / "attribute_spec.json").read_text(encoding="utf-8")).get("specs", [])
             if (folder / "attribute_spec.json").exists() else [])
    loaded = add_rule_set(app_config, job.document)
    logger.info("[JOBS] документ %s подгружен: групп %d, паспортов %d", job.document, len(groups), len(specs))
    return {"standard": job.document, "rule_groups": len(groups), "attribute_specs": len(specs),
            "loaded": loaded is not None}


jobs.on_extracted(_load_extracted)


@app.post("/process")
def process_excel(
        file: UploadFile = File(...),
        app_config: AppConfig = Depends(get_app_config),
):
    """POST /process: таблица Attribute/Value → результат .xlsx с колонкой каждого документа."""
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


_UNSAFE_NAME_RE = re.compile(r"[^\w.\- ]+", re.U)


def store_document(workspace, name: str, content: bytes):
    """Присланный PDF документа → папка входных документов извлечения и файл диалога.

    Имя файла — это имя будущего стандарта (папка в data/output). Файл с тем же именем и другим
    содержимым не затирается: новый получает суффикс.
    """
    stem = _UNSAFE_NAME_RE.sub("_", Path(name).stem).strip(" ._") or "document"
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    target, suffix = INPUT_DIR / f"{stem}.pdf", 2
    while target.exists() and target.read_bytes() != content:
        target, suffix = INPUT_DIR / f"{stem}_{suffix}.pdf", suffix + 1
    if not target.exists():
        target.write_bytes(content)
    return workspace.add(name=target.name, frame=pd.DataFrame(), kind="document", path=str(target),
                         origin="PDF нормативного документа")


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
    if uploaded is not None and uploaded.filename.lower().endswith(".pdf"):
        stored = store_document(workspace, uploaded.filename, uploaded.file.read())
        notes.append(f"The user has just attached the PDF document {stored.name} (file_id={stored.file_id}). "
                     f"It can be extracted into a new standard with extract_document.")
    elif uploaded is not None:
        try:
            stored = store_upload(workspace, uploaded.filename, uploaded.file.read())
        except Exception as e:
            logger.warning("[CHAT] файл %r не прочитан: %s", uploaded.filename, e)
            return {"reply": f"Не удалось прочитать «{uploaded.filename}». Нужен .xlsx с "
                             f"колонками Attribute и Value или PDF документа для извлечения.",
                    "session_id": session, "files": [], "jobs_active": len(jobs.active(session))}
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
    toolbox = Toolbox(workspace, app_config, catalogue(app_config), selected, produced, session_id=session)
    running = jobs.active(session)
    jobs_note = ("Background jobs of this conversation still running: "
                 + "; ".join(f"{j['title']} ({j['state']}, {j['step']}"
                             + (f", {j['cells_done']} of {j['cells_total']} cells done, {j['percent']}%"
                                if j.get("cells_total") else "")
                             + (f", pages read {j['pages_read']} of {j['pages_total']}" if j.get("pages_total") else "")
                             + ")" for j in running) + "."
                 if running else "")
    for note in (focus_note(workspace), toolbox.pending_note(), jobs_note):
        if note:
            notes.append(note)
    tools = build_tools(toolbox)
    prompt = f"[Context: {' '.join(notes)}]\n\n{text or 'Обработай приложенный файл.'}"
    try:
        reply = run_agent(app_config.agent_model or app_config.assistant or app_config.model, tools,
                          history_messages(_parse_json_list(history)), prompt)
    finally:
        save_workspace(session)

    files = [_file_payload(workspace.get(file_id)) for file_id in produced if workspace.get(file_id)]
    if toolbox.backgrounded and not files:
        # «запущено, выполнено N%» ничего не даёт: результат придёт отдельным сообщением с файлом
        reply = ""
    return {"reply": reply, "session_id": session, "files": files, "jobs_active": len(jobs.active(session))}


def _file_payload(stored) -> dict:
    """Файл рабочей области для ответа /chat: имя, тип и .xlsx в base64."""
    buffer = _write_excel_with_highlight(stored.frame, stored.ambiguous, stored.layout)
    return {"name": _download_name(stored), "media_type": XLSX_MEDIA_TYPE,
            "content_base64": base64.b64encode(buffer.read()).decode("ascii")}


def _job_message(job, workspace) -> dict:
    """Сообщение в чат о закончившейся фоновой задаче — из фактов задачи, без модели."""
    files, standards_changed = [], False
    took = job.finished - job.created if job.finished else 0
    minutes = (f" за {round(took / 60)} мин" if took >= 60 else f" за {round(took)} с") if took else ""
    if job.kind == "extract" and job.state == "done":
        r = job.result
        standards_changed = bool(r.get("loaded"))
        if r.get("rule_groups") == 0:
            # рассказ вместо стандарта: извлечение прошло, но правил в тексте нет
            text = (f"Документ «{job.document}» прочитан{minutes}, но правил нормализации в нём не найдено "
                    f"— стандарт из него не создан. Это точно нормативный документ?")
            return {"job_id": job.job_id, "message": text, "files": files, "standards_changed": False}
        text = (f"Документ «{job.document}» извлечён{minutes}: групп правил — {r.get('rule_groups', '?')}, "
                f"паспортов атрибутов — {r.get('attribute_specs', '?')}. "
                + ("Стандарт загружен — его можно выбрать в списке стандартов и нормализовать по нему файлы."
                   if standards_changed else f"Загрузить его в сервис не удалось: {r.get('load_error', '')}"))
    elif job.kind == "extract":
        tail = job.result.get("log_tail", "")
        text = (f"Извлечение «{job.document}» не удалось: {job.error}."
                + (f"\n\nПоследние строки журнала:\n```\n{tail}\n```" if tail else ""))
    elif job.state == "done":
        stored = workspace.get(job.result.get("file_id", ""))
        if stored is not None:
            files.append(_file_payload(stored))
        text = (f"Нормализация готова{minutes}: «{job.result.get('name', '')}», строк — {job.result.get('rows', '?')}, "
                f"жёлтых ячеек — {job.result.get('flagged', '?')}.")
    else:
        text = f"Задача «{job.title}» не выполнена: {job.error}."
    return {"job_id": job.job_id, "message": text, "files": files, "standards_changed": standards_changed}


@app.get("/jobs")
def list_jobs(session_id: str = ""):
    """Фоновые задачи диалога: идущие и закончившиеся, о которых пользователь ещё не знает.

    Фронтенд опрашивает, пока что-то идёт; закончившиеся отдаются один раз — сообщением с файлом.
    """
    session = (session_id or "").strip()
    if not session:
        return {"active": [], "finished": []}
    jobs.tick()
    workspace = get_workspace(session)
    return {"active": jobs.active(session),
            "finished": [_job_message(job, workspace) for job in jobs.take_finished(session)]}


# MCP: те же инструменты, что у агента чата, для внешних клиентов (см. mcp_server.py)
from .mcp_server import mcp  # noqa: E402

app.mount("/mcp", mcp.streamable_http_app())
