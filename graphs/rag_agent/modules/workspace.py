"""Файлы одного диалога: то, к чему агент может вернуться в следующем сообщении.

Рабочие области живут в памяти процесса и после каждого хода сохраняются в sqlite
(WORKSPACE_DB, по умолчанию graphs/rag_agent/data/workspaces.sqlite): перезапуск бэкенда — в том
числе автоперезапуск по --reload на каждой правке кода — файлы диалога больше не теряет.
Сохранённое старше SESSION_TTL_SECONDS удаляется.
"""

from __future__ import annotations

import logging
import os
import pickle
import sqlite3
import threading
import time
from collections import OrderedDict
from pathlib import Path
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger("normalization_pipeline")

# Границы, чтобы память не росла бесконечно: диалогов, файлов в диалоге и время жизни.
MAX_SESSIONS = 20
# Общий замок на выдачу номеров файлов (в объекте рабочей области он не сохранился бы в pickle).
_ADD_LOCK = threading.Lock()
MAX_FILES_PER_SESSION = 20
SESSION_TTL_SECONDS = 6 * 3600
DB_PATH = Path(os.getenv("WORKSPACE_DB", str(Path(__file__).resolve().parents[1] / "data" / "workspaces.sqlite")))


@dataclass
class StoredFile:
    """Таблица диалога. Байты не храним: xlsx рисуется из frame, когда его просят."""
    file_id: str
    name: str
    kind: str                     # "upload" — таблица пользователя, "result" — посчитали мы,
                                  # "document" — PDF нормативного документа (для извлечения)
    frame: pd.DataFrame
    ambiguous: dict[str, list[bool]] = field(default_factory=dict)
    origin: str = ""              # для результата: из чего и по каким стандартам посчитан
    # для результата: какие колонки исходные, какие — стандартов и их объяснений (results.layout_of)
    layout: dict = field(default_factory=dict)
    path: str = ""                # для документа: где лежит PDF
    created_at: float = field(default_factory=time.time)

    def summary(self) -> dict:
        return {
            "file_id": self.file_id,
            "name": self.name,
            "kind": self.kind,
            "rows": int(self.frame.shape[0]),
            "columns": [str(c) for c in self.frame.columns],
            "flagged_cells": sum(1 for flags in self.ambiguous.values() for f in flags if f),
            "origin": self.origin,
            **({"standards": [e["standard"] for e in self.layout.get("standards", [])]}
               if self.layout.get("standards") else {}),
            **({"path": getattr(self, "path", "")} if self.kind == "document" else {}),
        }


class Workspace:
    """Файлы одного диалога в порядке появления."""

    def __init__(self) -> None:
        self.files: OrderedDict[str, StoredFile] = OrderedDict()
        self.touched_at = time.time()
        self._counter = 0
        # номер сообщения пользователя: запись в базу подтверждается только в следующем сообщении
        self.turn = 0
        # отложенные записи в базу правил: токен → что записать (ждут «да» пользователя)
        self.pending: OrderedDict[str, dict] = OrderedDict()
        # о чём сейчас разговор: файл и строки, которые показали или правили последними. История
        # чата несёт только тексты реплик, без ответов инструментов, и без этого «эта строка» и
        # «тот файл» в следующем сообщении агенту уже не восстановить
        self.focus: dict = {"file_id": "", "rows": []}

    def add(self, name: str, frame: pd.DataFrame, kind: str = "upload",
            ambiguous: dict[str, list[bool]] | None = None, origin: str = "",
            layout: dict | None = None, path: str = "") -> StoredFile:
        with _ADD_LOCK:  # фоновая нормализация добавляет результат, пока идёт ход чата
            self._counter += 1
            file_id = f"f{self._counter}"
        stored = StoredFile(file_id=file_id, name=name, kind=kind, frame=frame,
                            ambiguous=ambiguous or {}, origin=origin, layout=layout or {}, path=path)
        self.files[stored.file_id] = stored
        while len(self.files) > MAX_FILES_PER_SESSION:
            dropped, _ = self.files.popitem(last=False)
            logger.info("[WORKSPACE] вытеснен файл %s (лимит %d)", dropped, MAX_FILES_PER_SESSION)
        self.touched_at = time.time()
        return stored

    def get(self, file_id: str) -> StoredFile | None:
        """Файл по id. Модель иногда называет файл по имени — это тоже принимаем."""
        self.touched_at = time.time()
        key = str(file_id).strip()
        if key in self.files:
            return self.files[key]
        for stored in reversed(self.files.values()):
            if stored.name.strip().lower() == key.lower():
                return stored
        return None

    def listing(self) -> list[dict]:
        self.touched_at = time.time()
        return [f.summary() for f in self.files.values()]

    def latest(self, kind: str = "upload") -> StoredFile | None:
        return next((f for f in reversed(self.files.values()) if f.kind == kind), None)


_SESSIONS: OrderedDict[str, Workspace] = OrderedDict()
_DB_LOCK = threading.Lock()


def _db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.execute("CREATE TABLE IF NOT EXISTS workspaces "
                       "(session_id TEXT PRIMARY KEY, touched_at REAL, data BLOB)")
    return connection


def save_workspace(session_id: str) -> None:
    """Сохраняет рабочую область диалога на диск (после хода). Сбой записи диалог не роняет."""
    workspace = _SESSIONS.get(session_id)
    if workspace is None:
        return
    try:
        blob = pickle.dumps(workspace, protocol=pickle.HIGHEST_PROTOCOL)
        with _DB_LOCK, _db() as connection:
            connection.execute("INSERT OR REPLACE INTO workspaces VALUES (?, ?, ?)",
                               (session_id, workspace.touched_at, blob))
    except Exception as e:
        logger.warning("[WORKSPACE] диалог %s не сохранён: %s", session_id, e)


def _load(session_id: str) -> Workspace | None:
    try:
        with _DB_LOCK, _db() as connection:
            row = connection.execute("SELECT touched_at, data FROM workspaces WHERE session_id = ?",
                                     (session_id,)).fetchone()
    except Exception as e:
        logger.warning("[WORKSPACE] диалог %s не прочитан: %s", session_id, e)
        return None
    if row is None or time.time() - row[0] > SESSION_TTL_SECONDS:
        return None
    try:
        # файл пишет только этот сервис (save_workspace) — данные свои, не присланные извне
        workspace = pickle.loads(row[1])
    except Exception as e:
        logger.warning("[WORKSPACE] диалог %s не восстановлен: %s", session_id, e)
        return None
    for name, default in (("turn", 0), ("pending", OrderedDict()), ("focus", {"file_id": "", "rows": []})):
        if not hasattr(workspace, name):
            setattr(workspace, name, default)  # сохранено старой версией класса
    logger.info("[WORKSPACE] диалог %s восстановлен с диска (%d файлов)", session_id, len(workspace.files))
    return workspace


def _evict() -> None:
    """Выбрасывает из памяти протухшие диалоги и самые старые сверх лимита; с диска — протухшие."""
    now = time.time()
    for key in [k for k, w in _SESSIONS.items() if now - w.touched_at > SESSION_TTL_SECONDS]:
        _SESSIONS.pop(key, None)
        logger.info("[WORKSPACE] диалог %s забыт по времени жизни", key)
    while len(_SESSIONS) > MAX_SESSIONS:
        key, _ = _SESSIONS.popitem(last=False)
        logger.info("[WORKSPACE] диалог %s вытеснен из памяти (лимит %d)", key, MAX_SESSIONS)
    try:
        with _DB_LOCK, _db() as connection:
            connection.execute("DELETE FROM workspaces WHERE touched_at < ?", (now - SESSION_TTL_SECONDS,))
    except Exception as e:
        logger.warning("[WORKSPACE] очистка диска не удалась: %s", e)


def get_workspace(session_id: str) -> Workspace:
    """Рабочая область диалога: из памяти, иначе с диска, иначе новая."""
    _evict()
    workspace = _SESSIONS.get(session_id)
    if workspace is None:
        workspace = _load(session_id) or Workspace()
        _SESSIONS[session_id] = workspace
    else:
        _SESSIONS.move_to_end(session_id)
    workspace.touched_at = time.time()
    return workspace


def focus_note(workspace: Workspace) -> str:
    """Фокус разговора строкой для контекста модели."""
    file_id = workspace.focus.get("file_id", "")
    stored = workspace.files.get(file_id) if file_id else None
    if stored is None:
        return ""
    rows = workspace.focus.get("rows") or []
    return (f"Current result: {stored.file_id} ({stored.name})."
            + (f" Rows the conversation was last about: {', '.join(map(str, rows))}." if rows else ""))
