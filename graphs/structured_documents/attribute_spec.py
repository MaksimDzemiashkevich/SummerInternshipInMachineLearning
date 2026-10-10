# -*- coding: utf-8 -*-
"""Прототип: спецификация атрибута — одна модель данных вместо ролей перечней, карточек и шаблонов.

Модель один раз читает правила документа и заполняет по каждому атрибуту одну спецификацию
(тип значения, список значений со всеми принятыми формами, единица, знаки, разделитель,
коэффициенты документа, ряд), а исполняет её код (graphs/rag_agent/spec_runner.py). Устроено по
образцу словарей свойств (IEC 61360, ECLASS): у свойства есть тип, формат, единица и список значений.
Результат — attribute_spec.json рядом с rule_groups.json.

    venv2/Scripts/python -m graphs.structured_documents.attribute_spec --doc <папка документа>
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from .modules.config_structured_documents_graph import Settings
from .rule_cards import GLOBAL_CHARS, _global_text, _invoke_json, _norm

load_dotenv()
settings = Settings()
OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "output"
BATCH = 6
KINDS = ("list", "quantity", "format", "text")


def _ground(spec: dict, text: str) -> None:
    """Значения и формы списка — только те, что буквально стоят в тексте документа."""
    if "values" not in spec:
        return
    values = []
    for entry in spec.get("values") or []:
        if not isinstance(entry, dict):
            continue
        value = str(entry.get("value") or "").strip()
        if not value or _norm(value) not in text:
            continue
        accepted = [a for a in (str(x).strip() for x in entry.get("accepted") or []) if a and _norm(a) in text]
        values.append({"value": value, "accepted": accepted})
    spec["values"] = values


MAX_CITED_TABLES = 6


def _cited_tables(group: dict, rule_groups: list[dict], tables: dict) -> str:
    """Таблицы документа, на которые ссылаются правила группы или строки других разделов об этом атрибуте.

    Принятые формы значения часто стоят в отдельной таблице, а группа атрибута и таблица класса
    изделия лишь ссылаются на неё («Written form: Canonical value of Table 12.41»). Без самой
    таблицы модель составляла спецификацию без этих форм, и «300#», «St6», «TMB» оставались как есть.
    """
    if not tables:
        return ""
    from graphs.rag_agent.modules.tables import table_references
    own = "\n".join(group.get("rules", []))
    name = str(group.get("attribute", "")).strip().lower()
    mentions = [line for g in rule_groups if g is not group for line in g.get("rules", [])
                if name and name in str(line).lower()]
    quoted = []
    for number in dict.fromkeys(table_references(own) + table_references("\n".join(mentions))):
        table = tables.get(number)
        if table is None or (table.title and table.title in own):
            continue
        quoted.append("\n".join(table.lines()))
        if len(quoted) >= MAX_CITED_TABLES:
            break
    return ("\n\nTables of the document cited for this attribute:\n" + "\n\n".join(quoted)) if quoted else ""


def compile_specs(rule_groups: list[dict], llm, doc_dir: Path | None = None) -> dict:
    global_text = _global_text(rule_groups)
    local = [g for g in rule_groups if g.get("type") != "global"
             and g.get("kind", "presentation") == "presentation"]
    tables = {}
    if doc_dir is not None:
        from graphs.rag_agent.modules.tables import load_tables
        tables = load_tables(doc_dir)
    texts = ["\n".join(g.get("rules", [])) + _cited_tables(g, rule_groups, tables) for g in local]

    def run(start: int) -> dict[int, dict]:
        chunk = local[start:start + BATCH]
        payload = {"GLOBAL": global_text[:GLOBAL_CHARS],
                   "groups": [{"id": start + i, "attribute": g.get("attribute", ""),
                               "rules": texts[start + i]} for i, g in enumerate(chunk)]}
        parsed = _invoke_json(llm, settings.ATTRIBUTE_SPEC_PROMPT, payload)
        return {s["id"]: s for s in parsed.get("specs", []) if isinstance(s, dict) and isinstance(s.get("id"), int)}

    with ThreadPoolExecutor(max_workers=5) as pool:
        by_id = {k: v for part in pool.map(run, range(0, len(local), BATCH)) for k, v in part.items()}

    specs = []
    for i, group in enumerate(local):
        spec = by_id.get(i)
        if not spec or spec.get("kind") not in KINDS:
            continue
        text = _norm(texts[i] + "\n" + global_text)
        _ground(spec, text)
        spec["variants"] = [v for v in spec.get("variants") or []
                            if isinstance(v, dict) and str(v.get("applies_to") or "").strip()]
        for variant in spec["variants"]:
            _ground(variant, text)
        spec["attribute"] = group.get("attribute", "")
        spec["aliases"] = list(group.get("aliases") or [])
        spec["rules"] = texts[i]  # с таблицами, на которые ссылаются правила
        spec.pop("id", None)
        specs.append(spec)
    return {"specs": specs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", required=True)
    args = parser.parse_args()
    llm = ChatOpenAI(model=settings.llmodel, max_tokens=16000, temperature=0, timeout=300,
                     model_kwargs={"response_format": {"type": "json_object"}})
    path = OUTPUT_DIR / args.doc / "rule_groups.json"
    specs = compile_specs(json.loads(path.read_text(encoding="utf-8")), llm, path.parent)
    (path.parent / "attribute_spec.json").write_text(json.dumps(specs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(args.doc, {k: sum(s["kind"] == k for s in specs["specs"]) for k in KINDS})


if __name__ == "__main__":
    main()
