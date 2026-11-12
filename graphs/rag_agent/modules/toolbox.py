"""Инструменты над файлами диалога и базой правил — один набор для агента чата и для MCP-сервера.

Каждый метод Toolbox — один инструмент: возвращает словарь (его увидит модель), а его docstring —
описание инструмента для модели, поэтому он по-английски. Агент чата (modules/agent.py) и MCP-сервер
(mcp_server.py) берут их отсюда, так что инструменты не расходятся между двумя входами.

Нормализованных значений здесь никто не придумывает: их считает конвейер, а сводки — код.
Запись в базу правил идёт в два шага: предпросмотр с токеном, затем подтверждение. В чате
подтверждение принимается только в следующем сообщении пользователя — модель не может сама
сказать за него «да» в том же ходе.
"""

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from ..pipeline.common import _attribute_matches, _norm_attr
from ..pipeline.frames import _row_context, prepare_frame
from . import jobs, results, rule_store
from .chat import AdHocRule
from .workspace import StoredFile, Workspace, save_workspace
from graphs.paths import DOCUMENTS_INPUT as INPUT_DIR, DOCUMENTS_OUTPUT as OUTPUT_DIR

logger = logging.getLogger("normalization_pipeline")

# Сколько строк показывает inspect_file за один раз.
INSPECT_LIMIT = 20
# Сколько строк файла пересчитывает предпросмотр правила.
PREVIEW_ROWS = 12
# Сколько сообщений пользователя ждёт подготовленная запись в базу: потом она снимается сама. Иначе
# забытое предложение (пользователь ответил «только на этот прогон», агент не отменил) висело в
# контексте каждого хода, и любое следующее «да» могло его записать
PENDING_TURNS = 3
# Сколько строк за раз можно пересчитать отдельно (дальше — normalize_file целиком).
MAX_RENORMALIZE_ROWS = 50
# Сколько ждать нормализацию в ходе чата: успела — файл в ответе, нет — придёт отдельным сообщением.
NORMALIZE_WAIT_SECONDS = 60
# Где лежат PDF документов для извлечения.


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


class Toolbox:
    """Инструменты одного диалога.

    selected — стандарты, выбранные в интерфейсе (пусто — все); produced — сюда складываются
    файлы, созданные или изменённые за этот ход (их вернут пользователю); require_new_turn —
    подтверждать запись в базу только в следующем сообщении (в чате да; MCP-клиент спрашивает
    пользователя сам).
    """

    TOOL_NAMES = ("list_files", "list_standards", "inspect_file", "list_values", "normalize_file",
                  "extract_document", "list_jobs",
                  "file_overview", "top_problems", "list_issues", "explain_row",
                  "renormalize_rows", "edit_result", "accept_proposals",
                  "list_rules", "propose_rule", "propose_rule_removal",
                  "confirm_pending", "cancel_pending")

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

    # ------------------------------------------------------------------ общее
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
        from ..pipeline.parsing import set_parser_model
        from ..spec_runner import get_runner
        set_parser_model(self._model)
        return get_runner(doc, self._model)

    # ------------------------------------------------------------------ файлы
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
        from ..pipeline.runner import normalize_table
        from ..pipeline.user_rules import user_rule_hits

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
        return {"status": "running", "job": jobs.describe(job),
                "message": "The normalization is still running in the background. Tell the user it is "
                           "running and that the result file will appear in the chat by itself when it is "
                           "ready; the chat stays free meanwhile. Do not wait or poll for it."}

    # ------------------------------------------------------------------ документы и фоновые задачи
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

    # ------------------------------------------------------------------ вопросы по результату
    def file_overview(self, file_id: str) -> dict:
        """Overview of a result file computed over all rows: which columns are source, context
        (e.g. Item) and standard columns; how many distinct items and attributes; per standard how
        many rows are flagged, changed and low_confidence, and the confidence levels.

        Metrics: "flagged" — the cell is yellow (the service is unsure or the document says to
        check it); "changed" — the recorded value did not conform to the standard (the result
        differs from it); "low_confidence" — the answer is not backed by a fact code checked
        (level medium, low, disagreed, not in list). Always tell the user which metric you used.
        """
        stored, error = self._result(file_id)
        if error:
            return error
        return {"file": stored.summary(),
                **results.overview(stored.frame, self._layout(stored), stored.ambiguous)}

    def top_problems(self, file_id: str, group_by: list[str] | None = None, metric: str = "flagged",
                     standard: str = "", top: int = 10, filters: dict | None = None) -> dict:
        """Rank groups of rows by how many problem rows they have — "which item has the most
        errors", "top 5 problem attributes of different items".

        `group_by`: column names to group by — a context column (as file_overview names it, e.g.
        "Item"), "Attribute", "standard" or "level"; several columns group by their combination.
        `metric`: "flagged" | "changed" | "low_confidence" (see file_overview). `standard`: one
        standard, or empty for all. `filters`: {column: text} keeps rows whose column contains the
        text; {column: [text, text]} — any of them ({"Attribute": ["Rated power", "Compressor input
        power"]}). Each group comes with count, share of its rows and example rows.
        """
        stored, error = self._result(file_id)
        if error:
            return error
        standard, error = self._column_standard(stored, standard)
        if error:
            return error
        return results.aggregate(stored.frame, self._layout(stored), stored.ambiguous,
                                 list(group_by or ["Attribute"]), metric, standard, top, filters)

    def list_issues(self, file_id: str, metric: str = "flagged", standard: str = "",
                    filters: dict | None = None, limit: int = 30) -> dict:
        """List the problem rows of a result and what is wrong with each: row number, attribute,
        recorded value, context, standard, result, confidence level, the note saying why (which
        rule or check), and for a flagged row the value(s) the service proposed.

        `metric`: "flagged" | "changed" | "low_confidence". `filters`: {column: text}, e.g.
        {"Item": "pump"} or {"Attribute": "Temperature"} or {"level": "disagreed"}; a list of texts
        matches any of them ({"Attribute": ["Rated power", "Compressor input power"]}).

        Levels: high — backed by a fact code checked (a form in the document's table, a quoted
        rule, a conversion done by code); medium — a model judgment confirmed by a second, blind
        model call; low — the model was unsure (a proposal is given); disagreed — two independent
        model calls gave different answers (NOT two standards disagreeing); not_in_list — the value
        is not one of the standard's listed values; document_flag — the document itself says to
        keep the value and pass it for review; not_covered — the standard does not govern this
        item; accepted / edited — a user accepted a proposal or corrected the cell; other — a
        standard without attribute specification (the note says what was applied).
        """
        stored, error = self._result(file_id)
        if error:
            return error
        standard, error = self._column_standard(stored, standard)
        if error:
            return error
        found = results.issues(stored.frame, self._layout(stored), stored.ambiguous,
                               metric, standard, filters, limit)
        if "rows" in found:
            self._focus(stored, [r["row"] for r in found["rows"]])
        return found

    def explain_row(self, file_id: str, row: int, standard: str = "") -> dict:
        """Explain one row of a result: what was recorded, what each standard wrote and why, which
        attribute specification of the standard the row was matched to (its kind, unit, number
        pattern, list values, instructions, the document's rules for it) and the rules users added
        to it. Use it when the user asks why a row came out the way it did.
        """
        from ..spec_runner import has_spec

        stored, error = self._result(file_id)
        if error:
            return error
        standard, error = self._column_standard(stored, standard)
        if error:
            return error
        positions, bad = self._positions(stored, [row])
        if not positions:
            return {"error": f"row {row} is outside 1..{stored.frame.shape[0]}"}
        position = positions[0]
        self._focus(stored, [position + 1])
        layout = self._layout(stored)
        found = [r for r in results.result_rows(stored.frame, layout, stored.ambiguous, standard)
                 if r["row"] == position + 1]
        source = self._source_frame(stored)
        attribute, value = str(source.iloc[position]["Attribute"]), str(source.iloc[position]["Value"])
        context = _row_context(source, position)
        explained = []
        for entry in found:
            rule_set = self._rule_set(entry["standard"])
            item = {k: entry[k] for k in ("standard", "result", "flagged", "level", "note", "proposals")}
            if rule_set is not None and has_spec(rule_set):
                try:
                    runner = self._spec_runner(rule_set.name)
                    spec = runner.spec_for(attribute, context, value)
                    if spec is not None:
                        spec = runner._variant(spec, context, value)
                        item["attribute_spec"] = {
                            "attribute": spec["attribute"], "variant": spec.get("applies_to", ""),
                            **{k: spec.get(k) for k in ("kind", "unit", "number_pattern", "locale",
                                                        "closed", "instructions", "flag_when")},
                            "values": [e["value"] for e in spec.get("values") or []][:20],
                            "rules": str(spec.get("rules", ""))[:1500],
                            "user_rules": rule_store.rules(rule_set.name, spec["attribute"])}
                    else:
                        item["attribute_spec"] = "the standard has no rules for this attribute; only general rules apply"
                except Exception as e:
                    item["attribute_spec_error"] = str(e)
            explained.append(item)
        return {"row": position + 1, "Attribute": attribute, "Value": value, "context": context,
                "standards": explained}

    # ------------------------------------------------------------------ правка результата
    def renormalize_rows(self, file_id: str, rows: list[int] | None = None, standards: list[str] | None = None,
                         attribute: str = "") -> dict:
        """Normalize again only these rows of a result file, with every rule of each standard —
        the document's rules together with the rules users stored in it — and write the new
        values into the same result. Use it after a rule was added, or when the user asks to redo
        particular rows. `rows` are row numbers as the other tools print them; `attribute` adds
        every row whose attribute label is this one (or an alias of it) — use it for "all rows of
        attribute X" instead of guessing row numbers; `standards` limits
        which standard columns are recomputed (empty = every standard of the file). Returns
        before → after for each row. The one-off rules column is not recomputed.
        """
        from ..pipeline.runner import run_rule_set

        stored, error = self._result(file_id)
        if error:
            return error
        rows = list(rows or [])
        if str(attribute or "").strip():
            source = self._source_frame(stored)
            rows += [i + 1 for i in range(source.shape[0])
                     if _attribute_matches(str(source.iloc[i]["Attribute"]), attribute)]
        positions, bad = self._positions(stored, rows)
        if not positions:
            return {"error": "no valid rows" + (f" and no rows of attribute {attribute!r}" if attribute else ""),
                    "rows_outside": bad, "file_rows": stored.frame.shape[0]}
        if len(positions) > MAX_RENORMALIZE_ROWS:
            return {"error": f"{len(positions)} rows asked; at most {MAX_RENORMALIZE_ROWS} at once — "
                             f"run normalize_file for the whole file instead"}
        layout = self._layout(stored)
        wanted = set()
        for name in standards or []:
            if not str(name).strip():
                continue
            resolved, error = self._column_standard(stored, name)
            if error:
                return error
            wanted.add(resolved)
        entries = [e for e in layout["standards"] if not wanted or e["standard"] in wanted]
        if not entries:
            return {"error": f"no such standard columns: {sorted(wanted)}",
                    "standards": [e["standard"] for e in layout["standards"]]}

        subset = self._source_frame(stored).iloc[positions].reset_index(drop=True)
        changes = []
        for entry in entries:
            rule_set = self._rule_set(entry["standard"])
            if rule_set is None:
                changes.append({"standard": entry["standard"], "error": "standard is not loaded"})
                continue
            column_df = run_rule_set(subset, self.app_config, rule_set)
            flags = stored.ambiguous.setdefault(entry["column"], [False] * stored.frame.shape[0])
            for i, position in enumerate(positions):
                index = stored.frame.index[position]
                before = stored.frame.at[index, entry["column"]]
                after = column_df.iloc[i]["Rules_Value"]
                note = str(column_df.iloc[i]["Applied"] or "")
                stored.frame.at[index, entry["column"]] = after
                if entry.get("applied") in stored.frame.columns:
                    stored.frame.at[index, entry["applied"]] = note
                flags[position] = bool(column_df.iloc[i]["Ambiguous"])
                changes.append({"row": position + 1, "standard": entry["standard"],
                                "Attribute": subset.iloc[i]["Attribute"], "Value": subset.iloc[i]["Value"],
                                "before": "" if pd.isna(before) else str(before), "after": str(after),
                                "flagged": flags[position], "note": note})
        logger.info("[AGENT] renormalize_rows %s | строки %s", stored.file_id, [p + 1 for p in positions])
        self._touch(stored)
        self._focus(stored, [p + 1 for p in positions])
        return {"file": stored.summary(), "changes": changes, "rows_outside": bad}

    def edit_result(self, file_id: str, row: int, column: str, value: str, flag: str = "keep") -> dict:
        """Change one cell of a result file, when the user explicitly asks for that correction.

        `row` is the row number as the tools print it (1 = first data row), `column` a column name
        of that file, `value` the text to put there. `flag` controls the review mark of the cell:
        "keep", "clear" (the user says it is fine) or "set". Only a result can be edited, never a
        file the user uploaded. Never use this to improve a result on your own: it overwrites what
        the rules produced. The answer carries `rule_hint` when the cell belongs to a standard:
        then offer to turn the correction into a rule of that standard (see propose_rule).
        """
        stored, error = self._file(file_id)
        if error:
            return error
        if stored.kind != "result":
            # правка исходника меняет то, из чего считаются все последующие прогоны, и
            # пользователь об этом не знает: свою таблицу он присылал как есть
            return {"error": f"file {stored.file_id} is the uploaded {stored.name}, not a "
                             f"result; uploads are never edited. Normalize it first, then "
                             f"correct the result."}
        columns = {str(c).strip().lower(): c for c in stored.frame.columns}
        target = columns.get(str(column).strip().lower())
        if target is None:
            return {"error": f"no column {column}", "columns": [str(c) for c in stored.frame.columns]}
        positions, _ = self._positions(stored, [row])
        if not positions:
            return {"error": f"row {row} is outside 1..{stored.frame.shape[0]}"}
        position = positions[0]
        index = stored.frame.index[position]

        before = "" if pd.isna(stored.frame.at[index, target]) else str(stored.frame.at[index, target])
        stored.frame.at[index, target] = value
        flags = stored.ambiguous.get(str(target))
        if flags is not None and position < len(flags) and flag in ("clear", "set"):
            flags[position] = flag == "set"
        entry = next((e for e in self._layout(stored)["standards"] if e["column"] == str(target)), None)
        report = {"file": stored.summary(),
                  "changed": {"row": position + 1, "column": str(target), "from": before, "to": value}}
        if entry is not None:
            if entry.get("applied") in stored.frame.columns:
                stored.frame.at[index, entry["applied"]] = f"исправлено вручную: было «{before}»"
            source = self._source_frame(stored)
            report["rule_hint"] = {
                "standard": entry["standard"], "attribute": str(source.iloc[position]["Attribute"]),
                "recorded": str(source.iloc[position]["Value"]), "service_wrote": before,
                "user_wrote": value, "context": _row_context(source, position)}
        logger.info("[AGENT] edit_result %s | строка %s, колонка %r: %r -> %r",
                    stored.file_id, row, str(target), before, value)
        self._touch(stored)
        self._focus(stored, [position + 1])
        return report

    def accept_proposals(self, file_id: str, standard: str = "", rows: list[int] | None = None,
                         filters: dict | None = None, choice: int = 0) -> dict:
        """Accept the values the service proposed for flagged rows, in one go, when the user asks
        ("accept all proposals for Temperature"). Only flagged rows that carry a proposal are
        touched (list_issues shows them). `rows` / `filters` ({column: text}) / `standard` select
        the rows. A row with two disagreeing proposals is skipped unless `choice` (1 or 2) says
        which one the user wants — ask them; never pick for them. The accepted cell loses its
        yellow mark.
        """
        stored, error = self._result(file_id)
        if error:
            return error
        standard, error = self._column_standard(stored, standard)
        if error:
            return error
        layout = self._layout(stored)
        wanted_rows = set(rows or [])
        accepted, skipped = [], []
        for r in results.result_rows(stored.frame, layout, stored.ambiguous, standard):
            if not r["flagged"] or (wanted_rows and r["row"] not in wanted_rows):
                continue
            if not results._matches(r, filters):
                continue
            if not r["proposals"]:
                if wanted_rows:
                    skipped.append({"row": r["row"], "standard": r["standard"], "reason": "no proposal"})
                continue
            if len(r["proposals"]) > 1 and int(choice or 0) not in (1, 2):
                # два разошедшихся ответа: какой из них верный, решает человек, а не порядок
                skipped.append({"row": r["row"], "standard": r["standard"],
                                "reason": "two proposals — ask which one", "proposals": r["proposals"]})
                continue
            pick = r["proposals"][min(max(1, int(choice or 1)), len(r["proposals"])) - 1]
            entry = next(e for e in layout["standards"] if e["standard"] == r["standard"])
            index = stored.frame.index[r["row"] - 1]
            stored.frame.at[index, entry["column"]] = pick
            if entry.get("applied") in stored.frame.columns:
                stored.frame.at[index, entry["applied"]] = f"принято предложение (пользователь): было «{r['result']}»; {r['note']}"
            stored.ambiguous.setdefault(entry["column"], [False] * stored.frame.shape[0])[r["row"] - 1] = False
            accepted.append({"row": r["row"], "standard": r["standard"], "Attribute": r["Attribute"],
                             "Value": r["Value"], "from": r["result"], "to": pick})
        if accepted:
            self._touch(stored)
        return {"file": stored.summary(), "accepted": accepted, "skipped": skipped}

    # ------------------------------------------------------------------ база правил
    def list_rules(self, standard: str = "", attribute: str = "") -> dict:
        """List the rules users stored in standards' knowledge bases (id, attribute, text, when).
        `standard` empty lists every standard; `attribute` narrows to one attribute.
        """
        names = [standard] if standard else [s["name"] for s in self.catalogue]
        found = []
        for name in names:
            rule_set = self._rule_set(name)
            if rule_set is None:
                return {"error": f"unknown standard {name}", "standards": [s["name"] for s in self.catalogue]}
            found += [{"standard": rule_set.name, **r} for r in rule_store.rules(rule_set.name, attribute)]
        return {"rules": found}

    def _standard_for_rules(self, standard: str):
        from ..spec_runner import has_spec
        rule_set = self._rule_set(standard)
        if rule_set is None:
            return None, {"error": "say which standard the rule is for" if not standard
                          else f"unknown standard {standard}",
                          "standards": [s["name"] for s in self.catalogue], "selected_in_ui": self.selected}
        if not has_spec(rule_set):
            return None, {"error": f"standard {rule_set.name} has no attribute specification "
                                   f"(attribute_spec.json), so rules cannot be stored in it yet. "
                                   f"Use a one-off rule with normalize_file instead."}
        return rule_set, None

    def _preview(self, runner, spec: dict | None, attribute: str, file_id: str,
                 rows: list[int] | None, standard: str) -> list[dict]:
        """Как легли бы строки файла с новым паспортом: строки этого атрибута и названные явно."""
        stored, error = self._file(file_id)
        if error or stored.kind != "result":
            return []
        if not rows and stored.file_id == self.workspace.focus.get("file_id"):
            rows = self.workspace.focus.get("rows") or []
        source = self._source_frame(stored)
        names = [attribute] + ([spec["attribute"]] + list(spec.get("aliases") or []) if spec else [])
        entry = next((e for e in self._layout(stored).get("standards", []) if e["standard"] == standard), None)
        matching = [p for p in range(source.shape[0])
                    if any(_attribute_matches(str(source.iloc[p]["Attribute"]), n) for n in names if n)]
        # строки разговора (фокус или переданные агентом) — только те, что этого атрибута: по имени или
        # потому, что их нормализовали по этому паспорту. Строки другого атрибута показывали в
        # предпросмотре не правило, а разброс модели (Ex marking «изменился» от правила про скорость)
        marker = f"паспорт «{spec['attribute']}»" if spec else ""
        explicit = [p for p in self._positions(stored, rows)[0]
                    if p in matching or (marker and entry is not None
                                         and marker in str(stored.frame.iloc[p].get(entry["applied"], "")))]
        positions = list(dict.fromkeys(explicit + matching))[:PREVIEW_ROWS]
        if not positions:
            return []
        trial = runner.clone_with(spec) if spec else runner

        def one(position: int) -> dict:
            label, value = str(source.iloc[position]["Attribute"]), str(source.iloc[position]["Value"])
            try:
                written, flag, note = trial.run(label, value, _row_context(source, position))
            except Exception as e:
                written, flag, note = value, True, f"ошибка: {e}"
            now, now_flagged = "", False
            if entry is not None:
                cell = stored.frame.iloc[position][entry["column"]]
                now = "" if pd.isna(cell) else str(cell)
                flags = stored.ambiguous.get(entry["column"]) or []
                now_flagged = bool(flags[position]) if position < len(flags) else False
            return {"row": position + 1, "Attribute": label, "Value": value, "now": now, "now_flagged": now_flagged,
                    "with_rule": written, "flagged": bool(flag), "note": note}

        with ThreadPoolExecutor(max_workers=6) as pool:
            return list(pool.map(one, positions))

    def _rows_to_redo(self, file_id: str, preview: list[dict], names: list[str]) -> list[int]:
        """Строки, которые после записи правила надо перепрогнать: те, что в предпросмотре меняются
        (значение или пометка), и строки атрибута, до которых предпросмотр не дошёл (PREVIEW_ROWS).

        Агент передавал в предпросмотр строки из фокуса разговора — строки прошлого правила, другого
        атрибута, — а после «да» перепрогонял их же, не тронув строки нового правила."""
        stored, error = self._file(file_id)
        if error or stored.kind != "result":
            return []
        changed = [p["row"] for p in preview
                   if " ".join(p["with_rule"].split()) != " ".join(p["now"].split()) or p["flagged"] != p.get("now_flagged")]
        seen = {p["row"] for p in preview}
        source = self._source_frame(stored)
        rest = [i + 1 for i in range(source.shape[0]) if i + 1 not in seen
                and any(_attribute_matches(str(source.iloc[i]["Attribute"]), n) for n in names if n)]
        return list(dict.fromkeys(changed + rest))

    def _pending(self, action: dict) -> str:
        token = uuid.uuid4().hex[:8]
        self.workspace.pending[token] = {**action, "turn": self.workspace.turn}
        while len(self.workspace.pending) > 10:
            self.workspace.pending.popitem(last=False)
        return token

    def propose_rule(self, standard: str, attribute: str, rule: str, create_new: bool = False,
                     file_id: str = "", rows: list[int] | None = None, item: str = "",
                     examples: list[dict] | None = None) -> dict:
        """Prepare a rule to be stored permanently in a standard's knowledge base (it then holds
        for every later run of that standard, together with the document's own rules, and wins
        where they conflict). Nothing is stored yet: the answer is a preview and a token.

        `standard`: the standard's name (ask the user when several are in play). `attribute`: the
        attribute as the standard or the file names it. `rule`: one imperative sentence in English
        that states the general rule (e.g. "Write the value with two decimals", "Write 'SS316' as
        '316 SS'"), not a one-cell fix. `file_id` and `rows`: the result the user is looking at —
        the preview then shows how those rows and the other rows of this attribute would come out.
        `item`: the kind of item the rule is only for (e.g. "gloves"), empty when it is for every
        item. `examples`: the user's own examples [{"value": "<recorded>", "expected": "<written>"}]
        — they are run through the new specification and the answer says which ones fail.

        The attribute's specification is compiled anew from the document's rules together with all
        user rules of the attribute (the same compiler as for documents), so whether the rule adds
        to or replaces a document rule, and what kind the attribute becomes, follows from its
        meaning; document sentences the rule contradicts are listed and hidden from the models.

        If the standard has no such attribute the answer is status "unknown_attribute" with similar
        attributes: tell the user, ask whether they meant one of those or want a NEW attribute, and
        only on their answer call again (with that attribute, or with create_new=true).
        Otherwise the answer is status "needs_confirmation": show the user the specification
        changes and the preview rows and ask; store it with confirm_pending only after the user
        agrees in a new message.
        """
        rule_set, error = self._standard_for_rules(standard)
        if error:
            return error
        text = str(rule or "").strip()
        if not text or not str(attribute or "").strip():
            return {"error": "both attribute and rule are needed"}
        runner = self._spec_runner(rule_set.name)
        spec = runner.spec_named(attribute)
        if spec is None:
            key = _norm_attr(attribute)
            by_alias = [s for s in runner.specs if key in {_norm_attr(a) for a in s.get("aliases") or []}]
            if len(by_alias) == 1 and not create_new:
                spec = by_alias[0]
        if spec is None and not create_new:
            return {"status": "unknown_attribute", "standard": rule_set.name, "attribute": attribute,
                    "similar": self._similar(runner, attribute),
                    "message": f"Standard {rule_set.name} has no attribute '{attribute}'. Ask the user "
                               f"whether they mean one of the similar attributes or want to create a "
                               f"new attribute with this rule."}
        name = spec["attribute"] if spec else str(attribute).strip()
        item = str(item or "").strip()
        new_text = f"For {item}: {text}" if item else text
        texts = [rule_store._rule_text(r) for r in rule_store.rules(rule_set.name, name)] + [new_text]
        new_examples = [{"value": str(e.get("value", "")).strip(), "expected": str(e.get("expected", "")).strip(), "item": item}
                        for e in examples or [] if isinstance(e, dict) and str(e.get("value", "")).strip()]
        all_examples = rule_store._examples(rule_store.rules(rule_set.name), name) + new_examples
        try:
            new_spec, notes, explanation = rule_store.spec_with_rules(rule_set.name, name, texts, spec, self._model,
                                                                      all_examples)
        except Exception as e:
            logger.warning("[RULES] паспорт %r не собран: %s", name, e)
            return {"error": f"could not build the specification with this rule: {e}"}
        if new_spec is None:
            return {"error": "the model returned no specification for this rule; rephrase it"}
        new_spec["attribute"] = name
        if spec is None:
            new_spec.setdefault("aliases", [])
        changes = rule_store.spec_changes(spec, new_spec)
        preview = self._preview(runner, new_spec, attribute, file_id, rows, rule_set.name)
        checks = rule_store.check_examples(rule_set.name, new_spec, all_examples, self._model)
        stored, _ = self._file(file_id)
        redo = self._rows_to_redo(file_id, preview, [attribute, name, *(new_spec.get("aliases") or [])])
        token = self._pending({"action": "add", "standard": rule_set.name, "attribute": name,
                               "rule": text, "item": item, "examples": new_examples,
                               "spec": new_spec, "new_attribute": spec is None,
                               "file_id": stored.file_id if stored is not None and stored.kind == "result" else "",
                               "rows": redo})
        return {"status": "needs_confirmation", "token": token, "standard": rule_set.name,
                "attribute": name, "new_attribute": spec is None, "rule": text,
                "spec_changes": changes, "notes": notes, "explanation": explanation,
                "preview": preview, "rows_changing": redo, "examples": checks, "regressions": new_spec.get("regressions") or [],
                "message": ("Nothing is stored yet. Show the user what changes and ask: store it in the "
                            "standard, or apply it only to this run? (Only this run: cancel_pending and pass "
                            "the rule to normalize_file as a one-off rule.) "
                            + ("WARNING: this creates a NEW attribute the standard does not define. "
                               if spec is None else "")
                            + ("The rule changes nothing in the specification: say so. " if not changes else "")
                            + ("Some of the user's examples FAIL with this specification: show them and say "
                               "the rule did not come out as stated. " if any(not c["ok"] for c in checks) else "")
                            + ("WARNING: values the rule does not mention would come out differently "
                               "(regressions: the document's own forms and examples, before → after). Show "
                               "them and ask whether that is intended. " if new_spec.get("regressions") else "")
                            + ("No preview row changes with this rule: tell the user so before asking, "
                               "and why if the notes say. " if preview and all(
                                   p["with_rule"] == p["now"] for p in preview) else ""))}

    def _similar(self, runner, attribute: str) -> list[str]:
        """Похожие атрибуты стандарта: выбор модели по смыслу и ближайшие по написанию."""
        from rapidfuzz import fuzz, process
        names = [s["attribute"] for s in runner.specs]
        similar = [n for n, score, _ in process.extract(attribute, names, scorer=fuzz.WRatio, limit=3) if score >= 60]
        try:
            chosen = runner.spec_for(attribute)
            if chosen is not None:
                similar = [chosen["attribute"]] + [n for n in similar if n != chosen["attribute"]]
        except Exception:
            pass
        return similar[:4]

    def propose_rule_removal(self, standard: str, rule_id: str, file_id: str = "") -> dict:
        """Prepare removing a stored rule (ids from list_rules). Nothing is removed yet: the answer
        is a preview and a token; remove it with confirm_pending after the user agrees.
        """
        rule_set, error = self._standard_for_rules(standard)
        if error:
            return error
        rule = next((r for r in rule_store.rules(rule_set.name) if r["id"] == rule_id), None)
        if rule is None:
            return {"error": f"no rule {rule_id} in {rule_set.name}", "rules": rule_store.rules(rule_set.name)}
        runner = self._spec_runner(rule_set.name)
        current = runner.spec_named(rule["attribute"])
        try:
            after, vanishes = rule_store.spec_without(rule_set.name, rule_id, self._model)
        except Exception as e:
            return {"error": f"could not rebuild the specification without this rule: {e}"}
        preview = [] if vanishes else self._preview(runner, after, rule["attribute"], file_id, None, rule_set.name)
        redo = self._rows_to_redo(file_id, preview, [rule["attribute"], *((current or {}).get("aliases") or [])])
        stored, _ = self._file(file_id)
        token = self._pending({"action": "remove", "standard": rule_set.name, "rule_id": rule_id,
                               "attribute": rule["attribute"], "spec": after, "rows": redo,
                               "file_id": stored.file_id if stored is not None and stored.kind == "result" else ""})
        return {"status": "needs_confirmation", "token": token, "rule": rule,
                "attribute_disappears": vanishes, "rows_changing": redo,
                "spec_changes": rule_store.spec_changes(current, after), "preview": preview,
                "message": "Nothing is removed yet. Show the user what changes and ask for a yes/no."}

    def confirm_pending(self, token: str) -> dict:
        """Carry out a prepared change of the knowledge base (from propose_rule or
        propose_rule_removal) — ONLY after the user agreed to it in their latest message.
        """
        from ..spec_runner import invalidate_runner
        self._expire_pending()
        action = self.workspace.pending.get(str(token).strip())
        if action is None:
            return {"error": f"no pending change {token}",
                    "pending": [{"token": t, "action": a["action"], "attribute": a["attribute"]}
                                for t, a in self.workspace.pending.items()]}
        if self.require_new_turn and action["turn"] >= self.workspace.turn:
            return {"error": "the user has not answered yet: show them the preview, ask, and wait for "
                             "their next message before confirming"}
        self.workspace.pending.pop(str(token).strip())
        if action["action"] == "add":
            # пользователь видел, что пример не сошёлся, и согласился сохранить правило как есть:
            # пример записывается с тем результатом, который он одобрил. Иначе неверный пример
            # проваливал бы каждую следующую сборку паспорта, и пересборка после переизвлечения
            # документа отвергалась бы навсегда
            failed = {str(c["value"]).strip(): c["got"] for c in action["spec"].get("examples_failed") or []}
            if failed:
                action["examples"] = [{**e, "expected": failed.get(str(e["value"]).strip(), e["expected"])}
                                      for e in action.get("examples") or []]
                action["spec"]["examples_failed"] = []
            rule = rule_store.add_rule(action["standard"], action["attribute"], action["rule"],
                                       action["spec"], action["new_attribute"], item=action.get("item", ""),
                                       examples=action.get("examples"))
            invalidate_runner(action["standard"])
            logger.info("[RULES] %s: правило %s для %r записано", action["standard"], rule["id"], action["attribute"])
            return {"stored": {"standard": action["standard"], **rule},
                    "rows_to_renormalize": action.get("rows") or [], "file_id": action.get("file_id", ""),
                    "message": "Stored. Offer to renormalize exactly rows_to_renormalize (the rows this rule "
                               "changes) with renormalize_rows — not other rows the conversation was about."}
        rule = rule_store.remove_rule(action["standard"], action["rule_id"], action["spec"])
        invalidate_runner(action["standard"])
        logger.info("[RULES] %s: правило %s снято", action["standard"], action["rule_id"])
        return {"removed": rule, "rows_to_renormalize": action.get("rows") or [], "file_id": action.get("file_id", ""),
                "message": "Removed. Offer to renormalize exactly rows_to_renormalize with renormalize_rows."}

    def cancel_pending(self, token: str) -> dict:
        """Drop a prepared change the user declined."""
        action = self.workspace.pending.pop(str(token).strip(), None)
        return {"cancelled": bool(action)}

    # ------------------------------------------------------------------ для чата
    def _expire_pending(self) -> None:
        for token in [t for t, a in self.workspace.pending.items()
                      if self.workspace.turn - a.get("turn", 0) > PENDING_TURNS]:
            self.workspace.pending.pop(token, None)

    def pending_note(self) -> str:
        """Отложенные записи в базу — строкой для контекста модели (токены живут в рабочей области)."""
        self._expire_pending()
        if not self.workspace.pending:
            return ""
        parts = []
        for token, a in self.workspace.pending.items():
            what = (f"add rule «{a['rule']}» to attribute {a['attribute']}" if a["action"] == "add"
                    else f"remove rule {a['rule_id']} of attribute {a['attribute']}")
            parts.append(f"token {token}: {what} in {a['standard']}")
        return ("Changes waiting for the user's yes/no (confirm_pending on yes, cancel_pending on no): "
                + "; ".join(parts) + ".")
