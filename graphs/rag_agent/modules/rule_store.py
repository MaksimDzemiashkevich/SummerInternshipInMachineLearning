"""Правила пользователей, записанные в базу документа: <папка документа>/user_rules.json.

«База» — это файлы извлечения документа (rule_groups.json, attribute_spec.json, …), а правила
людей лежат рядом отдельным файлом: повторное извлечение документа их не затирает, видно, что сказал
документ, а что добавили люди, и снять правило — удалить запись.

Паспорт атрибута с правилами пользователя собирается заново тем же компилятором, что и паспорта
документа (attribute_spec.compile_specs → spec_verify.verify_spec): правила документа для атрибута
плюс правила пользователя как последние поправки. Дополняет правило прежнее или отменяет его, какого
типа после этого атрибут, нужен ли вариант для особого вида значений — решает компилятор по смыслу,
как с поправками внутри самих документов. Строки документа, которые правило отменяет, отдельный вызов
называет дословно (SUPERSEDED_PROMPT); код находит их в тексте и убирает из текста правил этого
паспорта — иначе модели видят и правило, и отменённый им пример документа и спорят.

Первоисточник — тексты правил; собранный паспорт хранится рядом с отметкой, из какой редакции
документа он собран (basis). Переизвлекли документ — паспорт пересобирается, а не перебивает новый
паспорт документа устаревшей копией.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from .system_prompts import SUPERSEDED_PROMPT

logger = logging.getLogger("normalization_pipeline")

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "structured_documents" / "data" / "output"
USER_RULES_FILE = "user_rules.json"
# Пометка правила пользователя в тексте правил паспорта: ему отдан приоритет над документом.
USER_RULE_MARK = "[User rule — takes precedence over the document where they conflict]"
GLOBAL_CHARS = 60000

# Поля паспорта, которые показываются в предпросмотре изменений.
SPEC_FIELDS = ("kind", "closed", "unit", "number_pattern", "locale", "rounding", "rounding_scope",
               "unit_space", "number_format", "letter_case", "conversions", "series", "instructions",
               "flag_when", "values")

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


def _texts(rule_list: list[dict], attribute: str) -> list[str]:
    return [_rule_text(r) for r in rule_list if r["attribute"] == attribute]


def _examples(rule_list: list[dict], attribute: str) -> list[dict]:
    """Примеры правил атрибута: [{"value", "expected", "item"}] — ими проверяется собранный паспорт."""
    return [{**e, "item": r.get("item", "")} for r in rule_list if r["attribute"] == attribute
            for e in r.get("examples") or [] if str(e.get("value", "")).strip()]


def _rule_text(rule: dict) -> str:
    """Текст правила для компилятора: с изделием, если правило только для него."""
    return f"For {rule['item']}: {rule['text']}" if rule.get("item") else rule["text"]


# ---------------------------------------------------------------------- документ
def _document(doc: str) -> tuple[list[dict], dict[str, dict]]:
    """(группы правил документа, паспорта документа по имени атрибута) — без правил пользователей."""
    folder = OUTPUT_DIR / doc
    groups = json.loads((folder / "rule_groups.json").read_text(encoding="utf-8"))
    specs = json.loads((folder / "attribute_spec.json").read_text(encoding="utf-8"))["specs"]
    return groups, {s["attribute"]: s for s in specs}


def _global_text(groups: list[dict]) -> str:
    return "\n".join(r for g in groups if g.get("type") == "global" for r in g.get("rules", []))


def _group_for(groups: list[dict], attribute: str) -> dict | None:
    """Группа документа, из которой собран паспорт атрибута (паспорт носит имя группы)."""
    return next((g for g in groups if g.get("type") != "global"
                 and g.get("kind", "presentation") == "presentation"
                 and g.get("attribute", "") == attribute), None)


def _basis(groups: list[dict], attribute: str, texts: list[str]) -> str:
    """Отметка редакции: правила документа для атрибута, общие положения и тексты правил пользователя."""
    group = _group_for(groups, attribute)
    payload = json.dumps([group.get("rules") if group else None, _global_text(groups), texts], ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------- сборка паспорта
def _invoke_json(llm, prompt: str, payload: dict) -> dict:
    response = llm.invoke([SystemMessage(content=prompt),
                           HumanMessage(content=json.dumps(payload, ensure_ascii=False))],
                          response_format={"type": "json_object"})
    return json.loads(str(response.content))


def _norm(text: str) -> str:
    return " ".join(str(text).lower().split())


def _remove_superseded(spec: dict, texts: list[str], llm) -> list[str]:
    """Убирает из текста правил паспорта строки документа, которые правила пользователя отменяют.

    Строка, отменённая целиком, убирается; отменённая частично — остаётся с пометкой, какая её часть
    больше не действует: убрать её целиком значило бы потерять и то, что правило не трогало
    («semi-synthetic и synthetic blend → CHECK» при правиле только про semi-synthetic). Цитата
    принимается, только если она дословно стоит в тексте правил документа; возвращает описания.
    """
    rules_text = spec.get("rules", "")
    document_part = "\n".join(line for line in rules_text.splitlines() if not line.startswith(USER_RULE_MARK))
    try:
        answer = _invoke_json(llm, SUPERSEDED_PROMPT, {"rules": document_part[:20000], "user_rules": texts})
    except Exception as e:
        logger.warning("[RULES] отменённые строки не определены: %s", e)
        return []
    removed = []
    for item in answer.get("superseded") or []:
        if isinstance(item, str):
            item = {"quote": item, "whole": True}
        quote = str(item.get("quote") or "").strip()
        if len(quote) < 8 or quote not in document_part:
            continue  # без дословного совпадения ничего не трогаем
        if item.get("whole", True):
            rules_text = rules_text.replace(quote, "")
            removed.append(quote)
        else:
            part = str(item.get("overridden") or "").strip()
            rules_text = rules_text.replace(
                quote, f"{quote} [Partly superseded by the user rule — no longer holds: {part or 'what the user rule covers'}; the rest still holds.]")
            removed.append(f"{quote} (частично: {part})")
    spec["rules"] = rules_text
    return removed


# Сколько раз собирать паспорт, пока он не пройдёт примеры правил: компилятор недетерминирован —
# из одного правила «level n → Ln» он собрал то вариант-формат (верно), то буквальное значение «Ln»
# в закрытом списке (уверенная ошибка).
COMPILE_ATTEMPTS = 3


def check_examples(doc: str, spec: dict, examples: list[dict], llm) -> list[dict]:
    """Примеры через собранный паспорт (исполнитель без паспортов пользователей, с этим одним)."""
    if not examples:
        return []
    from ..pipeline.parsing import set_parser_model
    from ..spec_runner import SpecRunner
    set_parser_model(llm)
    trial = SpecRunner(doc, llm, user_rules=False).clone_with(spec)
    checks = []
    for example in examples:
        value, expected = str(example["value"]).strip(), str(example.get("expected", "")).strip()
        try:
            written, flag, note = trial.run(spec["attribute"], value,
                                            f"Item: {example['item']}" if example.get("item") else "")
        except Exception as e:
            written, flag, note = value, True, f"ошибка: {e}"
        checks.append({"value": value, "expected": expected, "got": written, "flagged": bool(flag),
                       "ok": not expected or " ".join(written.split()) == " ".join(expected.split()),
                       "note": note})
    return checks


# Сколько форм и примеров документа прогоняется через новый паспорт, чтобы увидеть, что правило
# задело то, чего не касалось
REGRESSION_FORMS = 40
REGRESSION_EXAMPLES = 12


def _mentioned(value: str, texts: list[str], examples: list[dict]) -> bool:
    """Правило говорит об этой форме: она в примерах пользователя или словом в тексте правила (без учёта
    регистра: «Write L as Large» говорит и о «large»)."""
    value = str(value or "").strip()
    if not value:
        return True
    if any(_norm(value) == _norm(e.get("value", "")) for e in examples):
        return True
    pattern = r"(?<!\w)" + re.escape(value) + r"(?!\w)"
    return any(re.search(pattern, t, re.IGNORECASE) for t in texts)


def _list_entries(spec: dict) -> list[dict]:
    """Значения списка паспорта и всех его вариантов."""
    entries = list(spec.get("values") or [])
    for variant in spec.get("variants") or []:
        entries += list(variant.get("values") or [])
    return entries


def restore_forms(current: dict | None, new: dict, texts: list[str], examples: list[dict]) -> list[str]:
    """Формы прежнего паспорта, которые компилятор выронил, хотя правило о них ничего не говорит,
    возвращаются к своему значению. Пересобирая паспорт ради «L → Large», модель потеряла «size 10 → 10»
    и «large → 9»; правило их не отменяло."""
    from ..spec_runner import form_key
    if not current:
        return []
    present = {form_key(f) for e in _list_entries(new) for f in [e["value"]] + list(e.get("accepted") or [])}
    restored = []
    for entry in _list_entries(current):
        for form in entry.get("accepted") or []:
            if form_key(form) in present or _mentioned(form, texts, examples):
                continue
            target = next((e for e in _list_entries(new) if e["value"] == entry["value"]), None)
            if target is None:
                continue  # значения больше нет — это покажет прогон форм
            target.setdefault("accepted", []).append(form)
            present.add(form_key(form))
            restored.append(f"{form} → {entry['value']}")
    return restored


def regression_probes(doc: str, current: dict | None, texts: list[str], examples: list[dict], llm) -> list[dict]:
    """Записи, на которых прежний паспорт работает и которых правило не касается, с его ответом:
    формы из списка и примеры документа (rule_cards), где прежний паспорт даёт ответ документа."""
    from ..pipeline.parsing import set_parser_model
    from ..spec_runner import SpecRunner
    if not current:
        return []
    forms = list(dict.fromkeys(f for e in _list_entries(current) for f in list(e.get("accepted") or [])))
    probes = [{"value": f, "expected": ""} for f in forms if not _mentioned(f, texts, examples)][:REGRESSION_FORMS]
    try:
        cards = json.loads((OUTPUT_DIR / doc / "rule_cards.json").read_text(encoding="utf-8")).get("groups", {})
    except (OSError, ValueError):
        cards = {}
    card_examples = [e for e in (cards.get(current["attribute"]) or {}).get("examples") or []
                     if str(e.get("from", "")).strip() and not _mentioned(e["from"], texts, examples)]
    probes += [{"value": str(e["from"]).strip(), "expected": str(e.get("to", "")).strip()}
               for e in card_examples[:REGRESSION_EXAMPLES]]
    probes = list({_norm(p["value"]): p for p in reversed(probes)}.values())[::-1]
    if not probes:
        return []
    set_parser_model(llm)
    reference = SpecRunner(doc, llm, user_rules=False).clone_with(current)
    kept = []
    for probe in probes:
        try:
            written, flag, _ = reference.run(current["attribute"], probe["value"], "")
        except Exception:
            continue
        # опора — только то, что прежний паспорт делает уверенно (и, для примера документа, верно)
        if flag or (probe["expected"] and _norm(written) != _norm(probe["expected"])):
            continue
        kept.append({"value": probe["value"], "before": written})
    return kept


def check_regressions(doc: str, spec: dict, probes: list[dict], llm) -> list[dict]:
    """Какие из этих записей новый паспорт пишет иначе или отправляет на проверку."""
    from ..spec_runner import SpecRunner
    if not probes:
        return []
    trial = SpecRunner(doc, llm, user_rules=False).clone_with(spec)
    changed = []
    for probe in probes:
        try:
            written, flag, _ = trial.run(spec["attribute"], probe["value"], "")
        except Exception as e:
            written, flag = f"ошибка: {e}", True
        if flag or _norm(written) != _norm(probe["before"]):
            changed.append({"value": probe["value"], "before": probe["before"], "after": written,
                            "flagged": bool(flag)})
    return changed


def compile_spec(doc: str, attribute: str, texts: list[str], llm,
                 examples: list[dict] | None = None, current: dict | None = None) -> tuple[dict | None, list[str]]:
    """Паспорт атрибута из правил документа и правил пользователя: (паспорт, заметки).

    Если у правил есть примеры, паспорт собирается до COMPILE_ATTEMPTS раз, пока все примеры не
    сойдутся; не сошлись и в последний раз — паспорт возвращается с "examples_failed", и решать,
    записывать ли его, будет тот, кто вызвал. Атрибута нет в документе — паспорт собирается только
    из правил пользователя и общих положений.

    `current` — паспорт, который работает сейчас. Тогда новый проверяется и на том, чего правило не
    касается: формы прежнего списка, выроненные компилятором, возвращаются кодом (restore_forms), а
    формы и примеры документа, которые теперь пишутся иначе, считаются как непрошедшие примеры
    ("regressions"): сборка повторяется, из попыток берётся лучшая, остаток видит пользователь.
    """
    examples = examples or []
    probes = regression_probes(doc, current, texts, examples, llm)
    best = None
    for attempt in range(COMPILE_ATTEMPTS if (examples or probes) else 1):
        spec, notes = _compile_once(doc, attribute, texts, llm)
        if spec is None:
            return None, notes
        restored = restore_forms(current, spec, texts, examples)
        if restored:
            notes.append("возвращены формы, которых правило не касается: " + "; ".join(restored[:10]))
        checks = check_examples(doc, spec, examples, llm)
        failed = [c for c in checks if not c["ok"]]
        regressions = check_regressions(doc, spec, probes, llm)
        spec["examples_failed"], spec["regressions"] = failed, regressions
        score = (len(failed), len(regressions))
        if best is None or score < best[2]:
            best = (spec, notes, score, attempt)
        if score == (0, 0):
            break
        logger.info("[RULES] %s/%s: сборка №%d — примеры %s, задеты %s", doc, attribute, attempt + 1,
                    [(c["value"], c["got"]) for c in failed], [(r["value"], r["after"]) for r in regressions])
    spec, notes, score, attempt = best
    if attempt and score == (0, 0):
        notes.append(f"примеры сошлись со сборки №{attempt + 1}")
    if spec["examples_failed"]:
        notes.append("примеры не сошлись: " + "; ".join(f"«{c['value']}» → «{c['got']}», ожидалось «{c['expected']}»"
                                                        for c in spec["examples_failed"]))
    if spec["regressions"]:
        notes.append("правило задело то, чего не касается: " + "; ".join(
            f"«{r['value']}»: было «{r['before']}», станет «{r['after']}»" + (" (на проверку)" if r["flagged"] else "")
            for r in spec["regressions"][:10]))
    return spec, notes


def _compile_once(doc: str, attribute: str, texts: list[str], llm) -> tuple[dict | None, list[str]]:
    """Одна сборка паспорта атрибута: компилятор, проверяющий, отменённые строки документа."""
    from graphs.structured_documents.attribute_spec import compile_specs
    from graphs.structured_documents.spec_verify import verify_spec

    groups, base_specs = _document(doc)
    group = _group_for(groups, attribute)
    group = dict(group) if group else {"type": "local", "kind": "presentation", "attribute": attribute,
                                       "aliases": [], "rules": []}
    group["rules"] = list(group.get("rules") or []) + [f"{USER_RULE_MARK} {t}" for t in texts]
    # компилятор и проверяющий ждут модель, которая отвечает JSON-объектом
    json_llm = llm.bind(response_format={"type": "json_object"})
    compiled = compile_specs([g for g in groups if g.get("type") == "global"] + [group],
                             json_llm, OUTPUT_DIR / doc)["specs"]
    if not compiled:
        return None, []
    spec, fixes = verify_spec(compiled[0], _global_text(groups), json_llm)
    notes = [f"проверка паспорта: {f['field']} — {f['problem'][:120]}" for f in fixes]
    removed = _remove_superseded(spec, texts, llm)
    notes += [f"отменена строка документа: «{q[:140]}»" for q in removed]
    base = base_specs.get(attribute)
    spec["attribute"] = attribute
    spec["aliases"] = list((base or {}).get("aliases") or group.get("aliases") or [])
    spec["user_rules"] = list(texts)
    spec["superseded"] = removed
    spec["new_attribute"] = base is None
    spec["user_basis"] = _basis(groups, attribute, texts)
    return spec, notes


def spec_with_rules(doc: str, attribute: str, texts: list[str], current: dict | None = None, llm=None,
                    examples: list[dict] | None = None) -> tuple[dict | None, list[str], str]:
    """Паспорт атрибута с этими правилами пользователя (все правила атрибута, не только новое)."""
    spec, notes = compile_spec(doc, attribute, texts, llm, examples, current)
    return spec, notes, ""


# ---------------------------------------------------------------------- запись и наложение
def add_rule(doc: str, attribute: str, text: str, spec: dict, new_attribute: bool,
             origin: str = "chat", item: str = "", examples: list[dict] | None = None) -> dict:
    """Записывает правило и паспорт атрибута, собранный со всеми его правилами."""
    with _lock:
        data = load(doc)
        number = 1 + max([int(str(r["id"]).lstrip("r")) for r in data["rules"]
                          if str(r.get("id", "")).lstrip("r").isdigit()] or [0])
        rule = {"id": f"r{number}", "attribute": attribute, "text": text, "item": item,
                "examples": [{"value": str(e.get("value", "")), "expected": str(e.get("expected", ""))}
                             for e in examples or [] if str(e.get("value", "")).strip()],
                "new_attribute": new_attribute, "origin": origin,
                "created_at": datetime.now().isoformat(timespec="seconds")}
        data["rules"].append(rule)
        data["specs"][attribute] = spec
        _save(doc, data)
    return rule


def remove_rule(doc: str, rule_id: str, spec: dict | None) -> dict | None:
    """Снимает правило; spec — паспорт, собранный с остальными правилами атрибута (None — их нет)."""
    with _lock:
        data = load(doc)
        rule = next((r for r in data["rules"] if r["id"] == rule_id), None)
        if rule is None:
            return None
        data["rules"] = [r for r in data["rules"] if r["id"] != rule_id]
        # у атрибута не осталось правил — паспорт документа снова его собственный
        if spec is None or not any(r["attribute"] == rule["attribute"] for r in data["rules"]):
            data["specs"].pop(rule["attribute"], None)
        else:
            data["specs"][rule["attribute"]] = spec
        _save(doc, data)
    return rule


def spec_without(doc: str, rule_id: str, llm) -> tuple[dict | None, bool]:
    """Паспорт атрибута после снятия правила: (паспорт или None, атрибут исчезает совсем)."""
    data = load(doc)
    rule = next((r for r in data["rules"] if r["id"] == rule_id), None)
    if rule is None:
        return None, False
    others = [r for r in data["rules"] if r["id"] != rule_id]
    remaining = _texts(others, rule["attribute"])
    _, base_specs = _document(doc)
    base = base_specs.get(rule["attribute"])
    if not remaining:
        return base, base is None
    spec, _ = compile_spec(doc, rule["attribute"], remaining, llm, _examples(others, rule["attribute"]))
    return spec, False


def apply_overrides(doc: str, specs: list[dict], model=None) -> list[dict]:
    """Паспорта документа с подставленными паспортами из user_rules.json.

    Паспорт, собранный по прежней редакции документа (документ переизвлекли), пересобирается, если
    есть модель; без модели — используется как есть, с предупреждением в лог.
    """
    data = load(doc)
    overrides = data["specs"]
    if not overrides:
        return specs
    groups = None
    for attribute, spec in list(overrides.items()):
        texts = _texts(data["rules"], attribute)
        groups = groups or _document(doc)[0]
        if spec.get("user_basis") == _basis(groups, attribute, texts):
            continue
        if model is None:
            logger.warning("[RULES] %s: паспорт %r собран по прежней редакции документа", doc, attribute)
            continue
        logger.info("[RULES] %s: документ изменился — паспорт %r пересобирается с правилами пользователя", doc, attribute)
        rebuilt, _ = compile_spec(doc, attribute, texts, model, _examples(data["rules"], attribute))
        if rebuilt is not None and rebuilt.get("examples_failed"):
            # новая сборка не проходит примеры правил — лучше прежний паспорт, чем уверенная ошибка
            logger.warning("[RULES] %s: пересобранный паспорт %r не проходит примеры — оставлен прежний", doc, attribute)
            continue
        if rebuilt is not None:
            with _lock:
                fresh = load(doc)
                fresh["specs"][attribute] = rebuilt
                _save(doc, fresh)
            overrides[attribute] = rebuilt
    merged = [overrides.get(s["attribute"], s) for s in specs]
    known = {s["attribute"] for s in specs}
    return merged + [s for name, s in overrides.items() if name not in known]


def _is_default(value) -> bool:
    return value in (None, "", [], False, "en", "half_up", "all", "decimal")


def _differs(left, right) -> bool:
    """Поле изменилось по существу: отсутствующее поле и его пустое значение — одно и то же."""
    return not (left == right or (_is_default(left) and _is_default(right)))


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
                # форма, совпадающая с самим значением, ничего не добавляет
                added = {a for a in set(entry.get("accepted") or []) - old.get(entry["value"], set())
                         if _norm(a) != _norm(entry["value"])}
                if entry["value"] not in old or added:
                    changes.append({"field": "values", "before": "", "after":
                                    f"{entry['value']} ← {', '.join(sorted(added)) or '(новое значение)'}"})
                lost = old.get(entry["value"], set()) - set(entry.get("accepted") or [])
                if lost:
                    changes.append({"field": "values", "before": f"{entry['value']} ← {', '.join(sorted(lost))}",
                                    "after": f"{entry['value']} (эти формы больше не принимаются)"})
            gone = set(old) - {e["value"] for e in after.get("values") or []}
            for value in sorted(gone):
                changes.append({"field": "values", "before": value, "after": "(убрано)"})
        elif _differs(before.get(field), after.get(field)):
            if not before and _is_default(after.get(field)):
                continue  # новый паспорт: поля по умолчанию пользователю ничего не говорят
            changes.append({"field": field, "before": before.get(field), "after": after.get(field)})
    old_variants = {v.get("applies_to"): v for v in before.get("variants") or []}
    for variant in after.get("variants") or []:
        old = old_variants.get(variant.get("applies_to"), {})
        for field in set(variant) | set(old):
            if field in ("applies_to", "variants") or not _differs(old.get(field), variant.get(field)):
                continue
            if not old and _is_default(variant.get(field)):
                continue  # новый вариант: показываем только то, что у него своё
            changes.append({"field": f"{field} [{variant.get('applies_to')}]",
                            "before": old.get(field), "after": variant.get(field, "(как у атрибута)")})
    for quote in after.get("superseded") or []:
        changes.append({"field": "rules", "before": quote[:200], "after": "(отменено правилом пользователя)"})
    return changes
