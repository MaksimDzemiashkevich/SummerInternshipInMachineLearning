"""Инструменты базы правил: сохранённые правила пользователя, их предпросмотр и запись в два шага
(предложение с токеном → подтверждение в следующем сообщении)."""
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from ...pipeline.common import _attribute_matches, _norm_attr
from ...pipeline.frames import _row_context
from .. import rule_store
from .base import ToolboxBase

logger = logging.getLogger("normalization_pipeline")

# Сколько строк файла пересчитывает предпросмотр правила.
PREVIEW_ROWS = 12
# Сколько сообщений пользователя ждёт подготовленная запись в базу: потом она снимается сама. Иначе
# забытое предложение (пользователь ответил «только на этот прогон», агент не отменил) висело в
# контексте каждого хода, и любое следующее «да» могло его записать
PENDING_TURNS = 3


class RuleTools(ToolboxBase):
    """Правила, которые пользователь сохраняет в стандарте, и подтверждение записи."""

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
        from ...spec_runner import has_spec
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
        from ...spec_runner import invalidate_runner
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
