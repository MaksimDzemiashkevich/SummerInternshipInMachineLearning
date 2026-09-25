"""Файлы одного диалога: то, к чему агент может вернуться в следующем сообщении.

Хранится в памяти процесса: перезапуск бэкенда (в том числе автоперезапуск по --reload)
диалоговые файлы теряет — это цена того, что на диск ничего не ложится и чистить нечего.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger("normalization_pipeline")

# Границы, чтобы память не росла бесконечно: диалогов, файлов в диалоге и время жизни.
MAX_SESSIONS = 20
MAX_FILES_PER_SESSION = 20
SESSION_TTL_SECONDS = 6 * 3600


@dataclass
class StoredFile:
    """Таблица диалога. Байты не храним: xlsx рисуется из frame, когда его просят."""
    file_id: str
    name: str
    kind: str                     # "upload" — прислал пользователь, "result" — посчитали мы
    frame: pd.DataFrame
    ambiguous: dict[str, list[bool]] = field(default_factory=dict)
    origin: str = ""              # для результата: из чего и по каким стандартам посчитан
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
        }


class Workspace:
    """Файлы одного диалога в порядке появления."""

    def __init__(self) -> None:
        self.files: OrderedDict[str, StoredFile] = OrderedDict()
        self.touched_at = time.time()
        self._counter = 0

    def add(self, name: str, frame: pd.DataFrame, kind: str = "upload",
            ambiguous: dict[str, list[bool]] | None = None, origin: str = "") -> StoredFile:
        self._counter += 1
        stored = StoredFile(file_id=f"f{self._counter}", name=name, kind=kind, frame=frame,
                            ambiguous=ambiguous or {}, origin=origin)
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


def _evict() -> None:
    """Выбрасывает протухшие диалоги и самые старые сверх лимита."""
    now = time.time()
    for key in [k for k, w in _SESSIONS.items() if now - w.touched_at > SESSION_TTL_SECONDS]:
        _SESSIONS.pop(key, None)
        logger.info("[WORKSPACE] диалог %s забыт по времени жизни", key)
    while len(_SESSIONS) > MAX_SESSIONS:
        key, _ = _SESSIONS.popitem(last=False)
        logger.info("[WORKSPACE] диалог %s вытеснен (лимит %d)", key, MAX_SESSIONS)


def get_workspace(session_id: str) -> Workspace:
    """Рабочая область диалога; неизвестный id заводит новую."""
    _evict()
    workspace = _SESSIONS.get(session_id)
    if workspace is None:
        workspace = Workspace()
        _SESSIONS[session_id] = workspace
    else:
        _SESSIONS.move_to_end(session_id)
    workspace.touched_at = time.time()
    return workspace
