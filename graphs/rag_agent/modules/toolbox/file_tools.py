"""Инструменты над файлами диалога и фоновыми задачами: список и просмотр файлов, нормализация
файла, извлечение документа из PDF, состояние задач."""
import logging
import uuid
from pathlib import Path

import pandas as pd

from graphs.paths import DOCUMENTS_INPUT as INPUT_DIR, DOCUMENTS_OUTPUT as OUTPUT_DIR

from .. import jobs, results
from ..chat import AdHocRule
from ..workspace import StoredFile, save_workspace
from .base import ToolboxBase

logger = logging.getLogger("normalization_pipeline")

# Сколько строк показывает inspect_file за один раз.
INSPECT_LIMIT = 20
# Сколько ждать нормализацию в ходе чата: успела — файл в ответе, нет — придёт отдельным сообщением.
NORMALIZE_WAIT_SECONDS = 60


def _parse_rules(raw) -> list[AdHocRule]:
    """Правила из аргументов инструмента: [{attribute, rule}] либо просто строки."""
    parsed: list[AdHocRule] = []
    for item in raw or []:
        if isinstance(item, dict):
            text = str(item.get("rule") or item.get("text") or "").strip()
            attribute = str(item.get("attribute") or "").strip()
            quantity = str(item.get("quantity") or "").strip()
        else:
            text, attribute, quantity = str(item).strip(), "", ""
        if text:
            parsed.append(AdHocRule(text=text, attribute=attribute, quantity=quantity))
    return parsed


def _rows_payload(stored: StoredFile, limit: int, contains: str) -> list[dict]:
    """Строки файла для осмотра, с пометкой, какие ячейки помечены на проверку."""
    frame = stored.frame
    rows = []
    for position in range(frame.shape[0]):
        row = {str(c): ("" if pd.isna(frame.iloc[position][c]) else str(frame.iloc[position][c]))
               for c in frame.columns}
        if contains and contains.lower() not in " ".join(row.values()).lower():
            continue
        flagged = [col for col, flags in stored.ambiguous.items()
                   if position < len(flags) and flags[position]]
        rows.append({"row": position + 1, **row,
                     **({"flagged_columns": flagged} if flagged else {})})
        if len(rows) >= max(1, limit):
            break
    return rows


class FileTools(ToolboxBase):
    """Файлы диалога, нормализация файла целиком, извлечение документов и фоновые задачи."""

    def list_files(self) -> dict:
        """List the files of this conversation: what the user uploaded and what was produced.

        Returns for each file its file_id (use it in the other tools), name, kind
        ("upload" or "result"), number of rows, columns, the standards of a result, and how
        many cells are flagged for human review.
        """
        return {"files": self.workspace.listing()}

    def list_standards(self) -> dict:
        """List the standards loaded into the service, with the name to pass to the other tools.

        `has_attribute_spec` says whether rules can be added to that standard's knowledge base.
        """
        loaded = {s["name"] for s in self.catalogue}
        return {"standards": [{**s, "has_attribute_spec": (OUTPUT_DIR / s["name"] / "attribute_spec.json").exists()}
                              for s in self.catalogue],
                "selected_in_ui": self.selected,
                "pdfs_not_extracted": sorted(p.name for p in INPUT_DIR.glob("*.pdf") if p.stem not in loaded)}

    def list_values(self, file_id: str, column: str = "Attribute", contains: str = "") -> dict:
        """Every distinct value of one column of a file with how many rows have it, most frequent
        first — the attribute labels (column "Attribute", the default), the items (a context
        column such as "Item"), or the values of a standard column.

        Use it to find rows by meaning: "how many rows are about power" — list the labels and pick
        yourself the ones that mean it ("Rated power", "Compressor input power"), in any language
        the user asks in. The file's labels are usually English, so a search for the user's own
        word misses them. `contains` keeps values containing that text (case-insensitive).
        """
        stored, error = self._file(file_id)
        if error:
            return error
        names = {str(c).strip().lower(): c for c in stored.frame.columns}
        key = names.get(str(column or "Attribute").strip().lower())
        if key is None:
            return {"error": f"no column {column!r}", "columns": [str(c) for c in stored.frame.columns]}
        values = stored.frame[key].fillna("").astype(str).str.strip()
        counts = values.value_counts()
        wanted = str(contains or "").strip().lower()
        listed = [{"value": v, "rows": int(n)} for v, n in counts.items() if not wanted or wanted in v.lower()]
        return {"file": stored.summary(), "column": str(key), "distinct": len(listed),
                "rows_total": int(stored.frame.shape[0]), "values": listed[:300],
                "truncated": len(listed) > 300}

    def inspect_file(self, file_id: str, limit: int = INSPECT_LIMIT, contains: str = "") -> dict:
        """Show rows of a file as they are, so you can answer questions about what is in it.

        `limit` caps how many rows come back; `contains` keeps only rows containing that text
        (case-insensitive). For a result file, each row also says which columns are flagged.
        For counting, ranking or finding problems use file_overview, top_problems and list_issues:
        they compute over the whole file, this tool only shows rows.
        """
        stored, error = self._file(file_id)
        if error:
            return error
        return {"file": stored.summary(), "rows": _rows_payload(stored, limit, contains)}

    def normalize_file(self, file_id: str, standards: list[str] | None = None,
                       rules: list[dict] | None = None) -> dict:
        """Normalize a whole file against the standards and return a new result file.

        `standards` are names from list_standards; leave it empty to use what the user picked
        in the interface. `rules` are one-off rules the user stated for this run (they are NOT
        stored in any standard), each {"attribute": "<the attribute it governs, or empty>",
        "quantity": "<a unit, or empty>", "rule": "<one imperative sentence in English>"}; they
        are applied in their own separate column and only in this run. Pass every one-off rule
        that should still hold, not just the newest. A rule about one attribute gets that
        attribute's name as written in the file. A rule about a kind of physical quantity
        wherever it occurs ("all masses in tonnes") gets "quantity": a unit of that kind ("kg")
        and an empty "attribute". A rule for every row gets both empty. The result says how many
        rows each rule matched. To store a rule in a standard permanently use propose_rule.
        """
        from ...pipeline.runner import normalize_table
        from ...pipeline.user_rules import user_rule_hits

        stored, error = self._file(file_id)
        if error:
            return error
        frame = self._source_frame(stored)
        missing = [c for c in ("Attribute", "Value") if c not in frame.columns]
        if missing:
            return {"error": f"file {stored.file_id} has no columns {missing}",
                    "columns": [str(c) for c in stored.frame.columns]}

        asked = [str(s).strip() for s in (standards or []) if str(s).strip()]
        # «RTE» вместо полного имени: однозначная часть имени — тот стандарт; неизвестное имя —
        # ошибка, а не молчаливый прогон по всем стандартам (33 колонки вместо одной)
        resolved = {name: self._rule_set(name) for name in asked}
        unknown = [name for name, rs in resolved.items() if rs is None]
        if unknown:
            return {"error": f"unknown or ambiguous standards {unknown}: nothing was run. Pass names "
                             f"from list_standards, or ask the user which standard they mean.",
                    "standards": [s["name"] for s in self.catalogue]}
        wanted = list(dict.fromkeys(rs.name for rs in resolved.values())) or self.selected
        parsed = _parse_rules(rules)
        logger.info("[AGENT] normalize_file %s | стандарты %s | правил %d",
                    stored.file_id, wanted or "все", len(parsed))

        by_name = {rs.name: rs for rs in self.app_config.rule_sets}
        rule_sets = [by_name[n] for n in wanted] or list(self.app_config.rule_sets)
        holder: dict = {}
        progress = jobs.Progress()  # «сколько осталось?» — list_jobs читает его по ходу

        def work() -> dict:
            result_df, ambiguous = normalize_table(frame, self.app_config, rule_sets, parsed, progress)
            layout = result_df.attrs.get("layout") or {}
            origin = (f"из {stored.name} по стандартам: {', '.join(wanted) if wanted else 'все'}"
                      + (f"; разовых правил: {len(parsed)}" if parsed else ""))
            result = self.workspace.add(name=f"result_{stored.name}", frame=result_df, kind="result",
                                        ambiguous=ambiguous, origin=origin, layout=layout)
            self.workspace.focus = {"file_id": result.file_id, "rows": []}
            report = {"result": result.summary(), "standards": wanted or "all",
                      "overview": results.overview(result.frame, self._layout(result), result.ambiguous)}
            if parsed:
                report["user_rules"] = [
                    {"rule": r.text, "attribute": r.attribute, "quantity": r.quantity, "rows_matched": n}
                    for r, n in zip(parsed, user_rule_hits(frame, parsed, self.app_config.model))]
            holder["report"] = report
            if self.session_id:
                save_workspace(self.session_id)
            return {"file_id": result.file_id, "name": result.name, "rows": int(result.frame.shape[0]),
                    "flagged": result.summary()["flagged_cells"]}

        # стандарты в названии: по нему агент потом рассказывает о задаче, и без них додумывал «по всем»
        title = f"нормализация {stored.name} по стандартам: {', '.join(wanted) if wanted else 'все загруженные'}"
        job = jobs.run_in_background(self.session_id, title, work, progress)
        if jobs.wait(job, NORMALIZE_WAIT_SECONDS):
            jobs.mark_delivered(job)  # файл уходит в этом же ответе
            if job.state != "done":
                return {"error": f"normalization failed: {job.error}"}
            result = self.workspace.get(job.result["file_id"])
            if result is not None:
                self._touch(result)
            return holder["report"]
        self.backgrounded.append(job.job_id)
        return {"status": "running", "job": jobs.describe(job),
                "message": "The normalization continues in the background and its file will appear in the "
                           "chat by itself. Say nothing about it. Do not wait or poll for it."}

    def extract_document(self, file_id: str = "", document: str = "") -> dict:
        """Extract a normalization standard from a PDF so it can be used to normalize files.

        `file_id`: a PDF the user attached in this conversation (list_files shows it as kind
        "document"); or `document`: the name of a PDF already in the service's input folder
        (list_standards shows them as pdfs_not_extracted). Extraction runs in the background and
        takes from minutes to an hour (a vision model reads every page, then rules are grouped and
        attribute specifications are compiled). Only one extraction runs at a time, others wait in a
        queue. The chat stays free meanwhile; when it is done the new standard is loaded and the user
        is told in the chat automatically. Tell the user that, and do not wait or poll.
        """
        pdf = None
        if str(file_id or "").strip():
            stored = self.workspace.get(file_id)
            if stored is None or stored.kind != "document":
                return {"error": f"{file_id} is not an attached PDF document", "files": self.workspace.listing()}
            pdf = Path(stored.path)
        elif str(document or "").strip():
            wanted = Path(str(document).strip()).stem.lower()
            pdfs = list(INPUT_DIR.glob("*.pdf"))
            exact = [p for p in pdfs if p.stem.lower() == wanted]
            partial = [p for p in pdfs if wanted in p.stem.lower()]
            pdf = exact[0] if exact else (partial[0] if len(partial) == 1 else None)
            if pdf is None:
                return {"error": f"no single PDF matches '{document}'",
                        "candidates": [p.name for p in partial] or sorted(p.name for p in pdfs)}
        else:
            return {"error": "give file_id of an attached PDF or the name of a PDF document"}
        if not pdf.exists():
            return {"error": f"file {pdf.name} is not on the server any more; ask the user to attach it again"}
        if (OUTPUT_DIR / pdf.stem / "rule_groups.json").exists():
            return {"error": f"document {pdf.stem} is already extracted and loaded as a standard", "standard": pdf.stem}
        same = [j for j in jobs._JOBS.values() if j.kind == "extract" and j.document == pdf.stem
                and j.state in ("queued", "running")]
        if same:
            return {"status": "already_running", "job": jobs.describe(same[0])}
        job = jobs.queue_extraction(self.session_id, pdf)
        logger.info("[AGENT] extract_document %s -> задача %s (%s)", pdf.name, job.job_id, job.state)
        waiting = [j for j in jobs._JOBS.values() if j.kind == "extract" and j.state in ("queued", "running")
                   and j.job_id != job.job_id]
        return {"status": job.state, "job": jobs.describe(job), "ahead_in_queue": len(waiting),
                "message": "Extraction runs in the background; the user will be told in the chat when it is "
                           "done, and the chat stays free meanwhile. Do not wait or poll."}

    def list_jobs(self) -> dict:
        """List the background jobs of this conversation (document extractions, long normalizations):
        state (queued, running, done, failed, lost), current step, errors; for an extraction the
        pages read so far, for a running normalization how many cells are done (cells_done of
        cells_total, percent; cells = rows × standards). Use it for "how much is left?" — answer
        in rows or percent from these numbers."""
        return {"jobs": [jobs.describe(j) for j in jobs.for_session(self.session_id)]}
