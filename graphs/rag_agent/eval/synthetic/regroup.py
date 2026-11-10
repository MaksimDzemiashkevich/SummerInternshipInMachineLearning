# -*- coding: utf-8 -*-
"""Перегруппировка готового data_after_check.json встроенным block_grouping в папку <doc>__v2.

Без VLM и без перезаписи штатной папки — для сравнения «было/стало» на одном и том же тексте.
    venv2/Scripts/python regroup.py <папка документа в output> [--window N] [--suffix __v3w]
--window N — раскладывать окнами по N символов (режим длинных документов), 0 — всегда одним вызовом;
по умолчанию — как в экстракции (block_grouping.WINDOW_CHARS).
"""
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]  # корень репозитория
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv(ROOT / ".env")

from graphs.structured_documents.block_grouping import WINDOW_CHARS, build_units, group_units
from graphs.structured_documents.modules.config_structured_documents_graph import Settings
from graphs.structured_documents.subject_tagging import tag_subjects

OUT = ROOT / "graphs/structured_documents/data/output"


def tail_helpers():
    """Функции хвоста экстракции без запуска графа (модуль графа исполняет его при импорте)."""
    source = (ROOT / "graphs/structured_documents/structured_documents_graph.py").read_text(encoding="utf-8")
    namespace: dict = {"re": re}
    exec("from __future__ import annotations\n"
         + source[source.index("_DET_RE = "):source.index("def distill_groups")], namespace)
    return namespace


def main():
    doc = sys.argv[1]
    args = sys.argv[2:]
    window = int(args[args.index("--window") + 1]) if "--window" in args else WINDOW_CHARS
    suffix = args[args.index("--suffix") + 1] if "--suffix" in args else "__v2"
    settings = Settings()
    src, dst = OUT / doc, OUT / (doc + suffix)
    pages = json.loads((src / "data_after_check.json").read_text(encoding="utf-8"))
    llm = ChatOpenAI(model=settings.llmodel, max_tokens=16000, temperature=0)
    units = build_units(pages)
    groups, lost = group_units(units, llm, settings.BLOCK_GROUPING_PROMPT, window or None)
    print(f"{doc}: блоков {len(units)}, групп {len(groups)}")
    for unit in lost:
        print("  НЕ НАЗНАЧЕН:", unit["id"], unit["text"][:120])

    h = tail_helpers()
    det, det_order, det_attrs = h["extract_deterministic_rules"](groups)
    phrase_map, phrase_attrs = h["extract_phrase_rules"](groups)
    files = {"deterministic_rules.json": det, "deterministic_order.json": det_order,
             "deterministic_attributes.json": det_attrs, "phrase_rules.json": phrase_map,
             "phrase_attributes.json": phrase_attrs,
             "excluded_attributes.json": h["extract_excluded_attributes"](groups),
             "render_templates.json": h["extract_render_templates"](groups)}
    groups = h["drop_empty_groups"](groups)
    files["rule_groups.json"] = groups
    files["subjects.json"] = tag_subjects(groups, llm)
    dst.mkdir(exist_ok=True)
    for name, obj in files.items():
        (dst / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy(src / "data_after_check.json", dst / "data_after_check.json")
    print("  предметы:", {k: v for k, v in files["subjects.json"].items() if v})
    for g in groups:
        print(f"  [{g['type']}/{g.get('kind')}] {g['attribute']} ({len(g['rules'])})")


if __name__ == "__main__":
    main()
