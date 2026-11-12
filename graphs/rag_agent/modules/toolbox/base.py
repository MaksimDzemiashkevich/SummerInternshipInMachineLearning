"""Основа инструментов диалога: файлы, стандарты и фокус разговора, общие для всех наборов инструментов."""
import logging

import pandas as pd

from ...pipeline.common import _norm_attr
from ...pipeline.frames import prepare_frame
from .. import results
from ..workspace import StoredFile, Workspace

logger = logging.getLogger("normalization_pipeline")


class ToolboxBase:
    """Состояние одного диалога и доступ к его файлам и стандартам.

    selected — стандарты, выбранные в интерфейсе (пусто — все); produced — сюда складываются
    файлы, созданные или изменённые за этот ход (их вернут пользователю); require_new_turn —
    подтверждать запись в базу только в следующем сообщении (в чате да; MCP-клиент спрашивает
    пользователя сам).
    """

    def __init__(self, workspace: Workspace, app_config, catalogue: list[dict],
                 selected: list[str] | None = None, produced: list[str] | None = None,
                 require_new_turn: bool = True, session_id: str = ""):
        self.workspace = workspace
        self.session_id = session_id
        self.app_config = app_config
        self.catalogue = catalogue
        self.selected = list(selected or [])
        self.produced = produced if produced is not None else []
        self.require_new_turn = require_new_turn

    @property
    def _model(self):
        # спецификации и правке паспорта нужна модель без привязанных инструментов
        return self.app_config.spec_model or self.app_config.assistant or self.app_config.model

    def _file(self, file_id: str) -> tuple[StoredFile | None, dict | None]:
        if not str(file_id or "").strip():
            # файл не назван — тот, о котором сейчас разговор, иначе последний результат
            file_id = (self.workspace.focus.get("file_id")
                       or getattr(self.workspace.latest("result"), "file_id", "")
                       or getattr(self.workspace.latest("upload"), "file_id", ""))
        stored = self.workspace.get(file_id)
        if stored is None:
            return None, {"error": f"no file {file_id}", "files": self.workspace.listing()}
        return stored, None

    def _layout(self, stored: StoredFile) -> dict:
        if not stored.layout:
            stored.layout = results.layout_of(stored.frame, {s["name"] for s in self.catalogue})
        return stored.layout

    def _result(self, file_id: str) -> tuple[StoredFile | None, dict | None]:
        stored, error = self._file(file_id)
        if error:
            return None, error
        if not self._layout(stored)["standards"]:
            return None, {"error": f"file {stored.file_id} ({stored.name}) has no standard columns: "
                                   f"it is not a normalization result. Normalize it first.",
                          "files": self.workspace.listing()}
        return stored, None

    def _rule_set(self, name: str):
        """Набор правил по имени стандарта: точно, без учёта регистра, по ярлыку или по части имени."""
        wanted = str(name or "").strip().lower()
        by_name = {rs.name: rs for rs in self.app_config.rule_sets}
        if not wanted:
            names = self.selected if len(self.selected) == 1 else []
            return by_name.get(names[0]) if names else None
        for entry in self.catalogue:
            if wanted in (entry["name"].lower(), entry["label"].lower()):
                return by_name.get(entry["name"])
        partial = [e["name"] for e in self.catalogue
                   if wanted in e["name"].lower() or wanted in e["label"].lower()]
        return by_name.get(partial[0]) if len(partial) == 1 else None

    def _column_standard(self, stored: StoredFile, standard: str) -> tuple[str | None, dict | None]:
        """Стандарт среди колонок результата: точно, без учёта регистра, по ярлыку или по
        однозначной части имени. Агент пишет «RTE»; фильтр по точному имени находил 0 строк, и
        агент отвечал «жёлтых нет» при трёх жёлтых."""
        wanted = str(standard or "").strip().lower()
        if not wanted:
            return None, None
        names = [e["standard"] for e in self._layout(stored)["standards"]]
        labels = {s["name"]: s["label"].lower() for s in self.catalogue}
        exact = [n for n in names if wanted in (n.lower(), labels.get(n, ""))]
        partial = [n for n in names if wanted in n.lower() or wanted in labels.get(n, "")]
        found = exact or partial
        if len(found) == 1:
            return found[0], None
        return None, {"error": f"standard {standard!r} is {'ambiguous' if found else 'not a column of this result'}",
                      "standards": names}

    def _source_frame(self, stored: StoredFile) -> pd.DataFrame:
        """Исходные колонки файла (без колонок стандартов) — то, из чего считаются значения."""
        columns = self._layout(stored)["source_columns"] or [str(c) for c in stored.frame.columns]
        return prepare_frame(stored.frame[columns].reset_index(drop=True))

    def _positions(self, stored: StoredFile, rows) -> tuple[list[int], list]:
        valid, bad = [], []
        for row in rows or []:
            try:
                position = int(row) - 1
            except (TypeError, ValueError):
                bad.append(row)
                continue
            (valid if 0 <= position < stored.frame.shape[0] else bad).append(
                position if 0 <= position < stored.frame.shape[0] else row)
        return list(dict.fromkeys(valid)), bad

    def _touch(self, stored: StoredFile) -> None:
        if stored.file_id not in self.produced:
            self.produced.append(stored.file_id)

    def _focus(self, stored: StoredFile, rows: list[int] | None = None) -> None:
        """Запомнить, о каком результате и каких строках сейчас разговор."""
        if stored.kind != "result":
            return
        if stored.file_id != self.workspace.focus.get("file_id"):
            self.workspace.focus = {"file_id": stored.file_id, "rows": []}
        if rows is not None:
            self.workspace.focus["rows"] = list(dict.fromkeys(int(r) for r in rows))[:20]

    def _spec_runner(self, doc: str):
        from ...pipeline.parsing import set_parser_model
        from ...spec_runner import get_runner
        set_parser_model(self._model)
        return get_runner(doc, self._model)
