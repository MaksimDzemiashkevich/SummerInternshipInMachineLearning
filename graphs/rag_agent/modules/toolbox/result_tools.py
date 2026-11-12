"""Инструменты вопросов о результате и его правки: сводка, проблемные строки, объяснение строки,
перепрогон строк, ручная правка, принятие предложенных значений."""
import logging

import pandas as pd

from ...pipeline.common import _attribute_matches
from ...pipeline.frames import _row_context
from .. import results, rule_store
from .base import ToolboxBase

logger = logging.getLogger("normalization_pipeline")

# Сколько строк за раз можно пересчитать отдельно (дальше — normalize_file целиком).
MAX_RENORMALIZE_ROWS = 50


class ResultTools(ToolboxBase):
    """Вопросы о результате нормализации и его правки — всё считается кодом по всем строкам."""

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
        from ...spec_runner import has_spec

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
        from ...pipeline.runner import run_rule_set

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
