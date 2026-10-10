"""Правила пользователей, записанные в базу документа: <папка документа>/user_rules.json.

«База» — это файлы извлечения документа (rule_groups.json, attribute_spec.json, …), а правила
людей лежат рядом отдельным файлом, а не правкой attribute_spec.json: повторное извлечение или
перекомпиляция документа их не затирает, видно, что сказал документ, а что добавили люди, и откат —
удаление записи.

Код исполняет поля паспорта атрибута (единицу, шаблон числа, пересчёты), а не текст правил, поэтому
просто дописанное «округлять до одного знака» на величину не подействовало бы. Правило превращается
в правку паспорта (RULE_PATCH_PROMPT): модель говорит, какие поля меняются, код проверяет их по схеме
и хранит здесь же получившийся паспорт. Паспорт нового атрибута компилируется с нуля тем же
attribute_spec.compile_specs, что и паспорта документа.
"""

from __future__ import annotations

import copy
import json
import threading
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from graphs.structured_documents.number_format import ROUNDING, valid_locale, valid_number_pattern

from .system_prompts import RULE_PATCH_PROMPT

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "structured_documents" / "data" / "output"
USER_RULES_FILE = "user_rules.json"
# Пометка правила пользователя в тексте правил паспорта: ему отдан приоритет над документом.
USER_RULE_MARK = "[User rule — takes precedence over the document where they conflict]"
GLOBAL_CHARS = 60000

# Поля паспорта, которые правка может менять, и проверка значения по схеме.
_SCALAR_CHECKS = {
    "kind": lambda v: v in ("list", "quantity", "format", "text"),
    "closed": lambda v: isinstance(v, bool),
    "unit": lambda v: isinstance(v, str),
    "number_pattern": lambda v: isinstance(v, str) and (v == "" or valid_number_pattern(v)),
    "locale": lambda v: isinstance(v, str) and valid_locale(v),
    "rounding": lambda v: v in ROUNDING,
    "rounding_scope": lambda v: v in ("all", "converted"),
    "unit_space": lambda v: isinstance(v, bool),
    "number_format": lambda v: v in ("decimal", "fraction"),
    "instructions": lambda v: isinstance(v, str),
    "flag_when": lambda v: isinstance(v, str),
    "letter_case": lambda v: v in ("", "upper", "lower"),
    "series": lambda v: isinstance(v, list) and all(isinstance(x, (int, float)) for x in v),
    "conversions": lambda v: isinstance(v, list) and all(
        isinstance(c, dict) and c.get("from") and isinstance(c.get("factor"), (int, float)) for c in v),
}
SPEC_FIELDS = tuple(_SCALAR_CHECKS) + ("values",)

_lock = threading.Lock()


# ---------------------------------------------------------------------- файл
def _path(doc: str) -> Path:
    return OUTPUT_DIR / doc / USER_RULES_FILE


def load(doc: str) -> dict:
    path = _path(doc)
    if not path.exists():
        return {"rules": [], "specs": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {"rules": [], "specs": {}}
    data.setdefault("rules", [])
    data.setdefault("specs", {})
    return data


def _save(doc: str, data: dict) -> None:
    path = _path(doc)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)  # запись целиком или никак: полузаписанный файл потерял бы все правила


def rules(doc: str, attribute: str = "") -> list[dict]:
    from ..pipeline.common import _norm_attr
    found = load(doc)["rules"]
    if attribute:
        found = [r for r in found if _norm_attr(r["attribute"]) == _norm_attr(attribute)]
    return found


def apply_overrides(doc: str, specs: list[dict]) -> list[dict]:
    """Паспорта документа с подставленными паспортами из user_rules.json."""
    overrides = load(doc)["specs"]
    if not overrides:
        return specs
    merged = [overrides.get(s["attribute"], s) for s in specs]
    known = {s["attribute"] for s in specs}
    return merged + [s for name, s in overrides.items() if name not in known]


def add_rule(doc: str, attribute: str, text: str, spec: dict, new_attribute: bool,
             origin: str = "chat") -> dict:
    """Записывает правило и паспорт атрибута, получившийся с ним."""
    with _lock:
        data = load(doc)
        number = 1 + max([int(str(r["id"]).lstrip("r") or 0) for r in data["rules"]
                          if str(r.get("id", "")).lstrip("r").isdigit()] or [0])
        rule = {"id": f"r{number}", "attribute": attribute, "text": text,
                "new_attribute": new_attribute, "origin": origin,
                "created_at": datetime.now().isoformat(timespec="seconds")}
        data["rules"].append(rule)
        data["specs"][attribute] = spec
        _save(doc, data)
    return rule


def remove_rule(doc: str, rule_id: str, spec: dict | None) -> dict | None:
    """Снимает правило; spec — паспорт атрибута без него (None — вернуть паспорт документа)."""
    with _lock:
        data = load(doc)
        rule = next((r for r in data["rules"] if r["id"] == rule_id), None)
        if rule is None:
            return None
        data["rules"] = [r for r in data["rules"] if r["id"] != rule_id]
        # у атрибута не осталось правил — паспорт документа снова его собственный: копия здесь
        # перебила бы паспорт, перекомпилированный потом из документа
        if spec is None or not any(r["attribute"] == rule["attribute"] for r in data["rules"]):
            data["specs"].pop(rule["attribute"], None)
        else:
            data["specs"][rule["attribute"]] = spec
        _save(doc, data)
    return rule


# ---------------------------------------------------------------------- паспорт с правилом
def _document(doc: str) -> tuple[list[dict], dict[str, dict]]:
    """(группы правил документа, паспорта документа по имени атрибута) — без правок пользователей."""
    folder = OUTPUT_DIR / doc
    groups = json.loads((folder / "rule_groups.json").read_text(encoding="utf-8"))
    specs = json.loads((folder / "attribute_spec.json").read_text(encoding="utf-8"))["specs"]
    return groups, {s["attribute"]: s for s in specs}


def _global_text(groups: list[dict]) -> str:
    return "\n".join(r for g in groups if g.get("type") == "global" for r in g.get("rules", []))


def _with_rule_text(rules_text: str, texts: list[str]) -> str:
    return rules_text + "".join(f"\n{USER_RULE_MARK} {t}" for t in texts)


def _invoke_json(llm, prompt: str, payload: dict) -> dict:
    response = llm.invoke([SystemMessage(content=prompt),
                           HumanMessage(content=json.dumps(payload, ensure_ascii=False))],
                          response_format={"type": "json_object"})
    return json.loads(str(response.content))


def _norm(text: str) -> str:
    return " ".join(str(text).lower().split())


def patch_spec(spec: dict, rule_texts: list[str], global_text: str, llm) -> tuple[dict, list[str], str]:
    """Паспорт с правилами пользователя: (новый паспорт, что поменялось, объяснение модели).

    Правки, не проходящие схему (шаблон числа с запятой, неизвестная локаль), отбрасываются и
    перечисляются — код не может исполнить то, что не по схеме. Принятые формы берутся, только
    если обе стороны дословно стоят в правиле пользователя или уже есть в паспорте: модель не
    может приписать пользователю соответствие, которого он не называл.
    """
    patched = copy.deepcopy(spec)
    payload = {"spec": {k: v for k, v in spec.items() if k in SPEC_FIELDS or k == "variants"},
               "rules": spec.get("rules", "")[:20000], "general_rules": global_text[:GLOBAL_CHARS],
               "user_rule": "\n".join(rule_texts)}
    answer = _invoke_json(llm, RULE_PATCH_PROMPT, payload)
    notes = []
    for change in answer.get("changes") or []:
        field, value = change.get("field"), change.get("value")
        check = _SCALAR_CHECKS.get(field)
        if check is None or not check(value):
            notes.append(f"отброшено (не по схеме): {field} = {value!r}")
            continue
        variant = str(change.get("variant") or "").strip()
        variants = [v for v in patched.get("variants") or [] if isinstance(v, dict)]
        if variant:
            target = next((v for v in variants if v.get("applies_to") == variant), None)
            if target is None:
                notes.append(f"отброшено (нет варианта «{variant}»): {field}")
                continue
            target[field] = value
            notes.append(f"{field} [{variant}]: {spec_field(spec, field, variant)!r} → {value!r}")
        else:
            before = patched.get(field)
            patched[field] = value
            for v in variants:  # правило для всех изделий: свои значения вариантов его не перебивают
                v.pop(field, None)
            notes.append(f"{field}: {before!r} → {value!r}")
    source = _norm("\n".join(rule_texts))
    known = {_norm(e["value"]) for e in patched.get("values") or []}
    for item in answer.get("add_forms") or []:
        value, form = str(item.get("value") or "").strip(), str(item.get("form") or "").strip()
        if not value or not form:
            continue
        if _norm(form) not in source or (_norm(value) not in source and _norm(value) not in known):
            notes.append(f"отброшено (нет дословно в правиле): {form} → {value}")
            continue
        values = patched.setdefault("values", [])
        entry = next((e for e in values if _norm(e["value"]) == _norm(value)), None)
        if entry is None:
            entry = {"value": value, "accepted": []}
            values.append(entry)
        if form not in entry["accepted"]:
            entry["accepted"].append(form)
        notes.append(f"форма: {form} → {value}")
    patched["rules"] = _with_rule_text(spec.get("rules", ""), rule_texts)
    patched["user_rules"] = list(spec.get("user_rules") or []) + list(rule_texts)
    return patched, notes, str(answer.get("explanation") or "")


def spec_field(spec: dict, field: str, variant: str = ""):
    if variant:
        target = next((v for v in spec.get("variants") or [] if v.get("applies_to") == variant), {})
        return target.get(field, spec.get(field))
    return spec.get(field)


def compile_new(doc: str, attribute: str, rule_texts: list[str], llm) -> dict | None:
    """Паспорт атрибута, которого в документе нет: только из правил пользователя и общих положений."""
    from graphs.structured_documents.attribute_spec import compile_specs
    groups, _ = _document(doc)
    group = {"type": "local", "kind": "presentation", "attribute": attribute, "aliases": [],
             "rules": [f"{USER_RULE_MARK} {t}" for t in rule_texts]}
    # compile_specs ждёт модель, которая уже отвечает JSON-объектом
    json_llm = llm.bind(response_format={"type": "json_object"})
    compiled = compile_specs([g for g in groups if g.get("type") == "global"] + [group],
                             json_llm, OUTPUT_DIR / doc)["specs"]
    if not compiled:
        return None
    spec = compiled[0]
    spec["user_rules"] = list(rule_texts)
    spec["new_attribute"] = True
    return spec


def spec_with_rules(doc: str, attribute: str, rule_texts: list[str], current: dict | None, llm
                    ) -> tuple[dict | None, list[str], str]:
    """Паспорт атрибута с правилами пользователя поверх current (None — новый атрибут)."""
    groups, _ = _document(doc)
    if current is None:
        spec = compile_new(doc, attribute, rule_texts, llm)
        return spec, (["новый паспорт атрибута"] if spec else []), ""
    return patch_spec(current, rule_texts, _global_text(groups), llm)


def spec_without(doc: str, rule_id: str, llm) -> tuple[dict | None, bool]:
    """Паспорт атрибута после снятия правила: (паспорт или None, атрибут исчезает совсем).

    Остальные правила этого атрибута накладываются заново на паспорт документа: снять одну правку
    из середины цепочки правок нельзя — поля могли поменяться несколько раз.
    """
    data = load(doc)
    rule = next((r for r in data["rules"] if r["id"] == rule_id), None)
    if rule is None:
        return None, False
    remaining = [r["text"] for r in data["rules"] if r["attribute"] == rule["attribute"] and r["id"] != rule_id]
    _, base_specs = _document(doc)
    base = base_specs.get(rule["attribute"])
    if not remaining:
        return base, base is None
    spec, _, _ = spec_with_rules(doc, rule["attribute"], remaining, base, llm)
    return spec, False


def spec_changes(before: dict | None, after: dict | None) -> list[dict]:
    """Какие поля паспорта отличаются — для предпросмотра пользователю."""
    if after is None:
        return [{"field": "*", "before": "паспорт атрибута", "after": "удалён"}]
    before = before or {}
    changes = []
    for field in SPEC_FIELDS:
        if field == "values":
            old = {e["value"]: set(e.get("accepted") or []) for e in before.get("values") or []}
            for entry in after.get("values") or []:
                added = set(entry.get("accepted") or []) - old.get(entry["value"], set())
                if entry["value"] not in old or added:
                    changes.append({"field": "values", "before": "", "after":
                                    f"{entry['value']} ← {', '.join(sorted(added)) or '(новое значение)'}"})
        elif before.get(field) != after.get(field):
            if not before and after.get(field) in (None, "", [], False, "en", "half_up", "all", "decimal"):
                continue  # новый паспорт: поля по умолчанию пользователю ничего не говорят
            changes.append({"field": field, "before": before.get(field), "after": after.get(field)})
    for variant in after.get("variants") or []:
        old = next((v for v in before.get("variants") or [] if v.get("applies_to") == variant.get("applies_to")), {})
        for field in set(variant) | set(old):
            if field != "applies_to" and variant.get(field) != old.get(field):
                changes.append({"field": f"{field} [{variant.get('applies_to')}]",
                                "before": old.get(field), "after": variant.get(field, "(как у атрибута)")})
    return changes
