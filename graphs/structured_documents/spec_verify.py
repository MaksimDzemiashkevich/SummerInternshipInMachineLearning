# -*- coding: utf-8 -*-
"""Независимая проверка спецификаций атрибутов (attribute_spec.json) другой моделью.

Спецификацию исполняет код, и её ошибка становится уверенной ошибкой в каждой строке: модель
переворачивала, что пишется и что лишь принимается на входе, ставила в поле единицы название
единицы словом, теряла условие, при котором действует число знаков. Здесь другая модель (по
умолчанию та же модель, что составляла, — отдельным вызовом в роли проверяющего, без своего ответа; другую можно задать в SPEC_VERIFY_MODEL) сверяет каждую спецификацию с правилами.
Исправление принимается, только если проверяющий дословно процитировал фразу правил и код нашёл
её в тексте; значения списка снова сверяются с текстом. Всё принятое записывается в поле
"verified_fixes" спецификации — видно, что и почему поменялось.

    venv2/Scripts/python -m graphs.structured_documents.spec_verify --doc <папка документа>
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from .attribute_spec import KINDS, _ground
from .number_format import ROUNDING, valid_locale, valid_number_pattern
from .modules.config_structured_documents_graph import Settings
from .rule_cards import GLOBAL_CHARS, _global_text, _invoke_json, _norm
from graphs.paths import DOCUMENTS_OUTPUT as OUTPUT_DIR

load_dotenv()
settings = Settings()
VERIFY_MODEL = os.getenv("SPEC_VERIFY_MODEL", settings.llmodel)  # та же модель, отдельный вызов в роли проверяющего
FIELDS = {"kind", "values", "closed", "unit", "number_pattern", "locale", "rounding", "rounding_scope", "unit_space",
          "number_format", "conversions", "series", "instructions", "flag_when", "letter_case", "variants"}
SKIP = {"rules", "attribute", "aliases", "verified_fixes"}


def _fits_schema(field: str, value) -> bool:
    """Исправленное значение поля — в схеме спецификации: её поля исполняет код, а чужие поля
    («from_unit», «condition») он не понимает. Проверяющий их придумывал."""
    is_list_of = lambda keys: isinstance(value, list) and all(isinstance(x, dict) and keys <= set(x) for x in value)
    if field == "kind":
        return value in KINDS
    if field == "values":
        return is_list_of({"value", "accepted"})
    if field == "conversions":
        return is_list_of({"from", "factor"})
    if field == "variants":
        return is_list_of({"applies_to"})
    if field == "series":
        return isinstance(value, list) and all(isinstance(x, (int, float)) for x in value)
    if field == "number_pattern":
        return valid_number_pattern(value)
    if field == "locale":
        return valid_locale(value)
    if field == "rounding":
        return value in ROUNDING
    if field == "rounding_scope":
        return value in ("all", "converted")
    if field in ("closed", "unit_space"):
        return isinstance(value, bool)
    if field == "number_format":
        return value in ("decimal", "fraction")
    if field == "letter_case":
        return value in ("", "upper", "lower")
    return isinstance(value, str)


def verify_spec(spec: dict, global_text: str, llm) -> tuple[dict, list[dict]]:
    """(исправленная спецификация, принятые исправления)."""
    payload = {"rules": spec["rules"], "general_rules": global_text[:GLOBAL_CHARS],
               "spec": {k: v for k, v in spec.items() if k not in SKIP}}
    parsed = None
    for attempt in range(3):
        try:
            parsed = _invoke_json(llm, settings.SPEC_VERIFY_PROMPT, payload)
            break
        except Exception as e:
            print(f"  проверка {spec['attribute']!r}: {type(e).__name__}, повтор")
            time.sleep(5 * (attempt + 1))
    if not parsed:
        return spec, []
    source = _norm(spec["rules"] + "\n" + global_text)
    accepted = []
    fixed = json.loads(json.dumps(spec))
    for issue in parsed.get("issues") or []:
        field, quote = issue.get("field"), _norm(issue.get("quote") or "")
        if field not in FIELDS or len(quote) < 8 or quote not in source:
            continue  # без дословной цитаты из правил исправление не принимается
        if not _fits_schema(field, issue.get("value")):
            continue  # исправление в своей, не в нашей схеме (код его не исполнит)
        variant_name = str(issue.get("variant") or "").strip()
        target = fixed
        if variant_name:
            target = next((v for v in fixed.get("variants") or [] if v.get("applies_to") == variant_name), None)
            if target is None:
                continue
        target[field] = issue.get("value")
        accepted.append({"field": field, "variant": variant_name, "problem": issue.get("problem", ""),
                         "quote": issue.get("quote", "")})
    _ground(fixed, source)
    for variant in fixed.get("variants") or []:
        if isinstance(variant, dict):
            _ground(variant, source)
    for target in [fixed] + [v for v in fixed.get("variants") or [] if isinstance(v, dict)]:
        fix = _check_zeros(fixed, target, global_text, source, llm)
        if fix:
            accepted.append(fix)
    fixed["verified_fixes"] = accepted
    return fixed, accepted


def _check_zeros(spec: dict, target: dict, global_text: str, source: str, llm) -> dict | None:
    """Узкий вопрос о шаблоне числа: дописывать нули до N знаков или округлять до N и убирать их.
    В общем вызове составления и проверки модель путала одно с другим («156 kg» → «156.0 kg»);
    на этот один вопрос она отвечает с цитатой, и шаблон меняется, только если цитата есть в тексте."""
    pattern = target.get("number_pattern")
    if not isinstance(pattern, str) or "." not in pattern or "@" in pattern:
        return None
    whole, _, fraction = pattern.partition(".")
    payload = {"rules": spec["rules"], "general_rules": global_text[:GLOBAL_CHARS],
               "unit": target.get("unit") or spec.get("unit") or "", "decimals": len(fraction)}
    try:
        parsed = _invoke_json(llm, settings.ZEROS_CHECK_PROMPT, payload)
    except Exception:
        return None
    zeros, quote = parsed.get("zeros"), str(parsed.get("quote") or "")
    if zeros not in ("show", "remove") or len(_norm(quote)) < 8 or _norm(quote) not in source:
        return None
    wanted = whole + "." + ("0" if zeros == "show" else "#") * len(fraction)
    if wanted == pattern:
        return None
    target["number_pattern"] = wanted
    return {"field": "number_pattern", "variant": str(target.get("applies_to") or ""),
            "problem": f"{pattern} -> {wanted} (нули в конце: {zeros})", "quote": quote}


def verify_doc(doc: str) -> None:
    """Проверяет все паспорта документа по его правилам и записывает принятые исправления."""
    folder = OUTPUT_DIR / doc
    data = json.loads((folder / "attribute_spec.json").read_text(encoding="utf-8"))
    global_text = _global_text(json.loads((folder / "rule_groups.json").read_text(encoding="utf-8")))
    llm = ChatOpenAI(model=VERIFY_MODEL, timeout=300, model_kwargs={"response_format": {"type": "json_object"}})
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda s: verify_spec(s, global_text, llm), data["specs"]))
    data["specs"] = [spec for spec, _ in results]
    (folder / "attribute_spec.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    fixes = [(spec["attribute"], f) for spec, accepted in results for f in accepted]
    print(f"{doc}: спецификаций {len(results)}, исправлено полей {len(fixes)} (модель {VERIFY_MODEL})")
    for attribute, f in fixes:
        where = f" [{f['variant']}]" if f["variant"] else ""
        print(f"  {attribute}{where}.{f['field']}: {f['problem'][:140]}")


def main() -> None:
    """CLI: проверка паспортов документа --doc."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", required=True)
    verify_doc(parser.parse_args().doc)


if __name__ == "__main__":
    main()
