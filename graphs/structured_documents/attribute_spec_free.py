# -*- coding: utf-8 -*-
"""Эксперимент C: спецификация атрибута со свободными полями — структуру выбирает модель.

В attribute_spec.py поля спецификации заданы заранее, потому что их исполняет код. Здесь модель
сама решает, какие поля нужны каждому атрибуту, а применяет описание тоже модель
(graphs/rag_agent/llm_runner.py, режим «free»). Вход тот же, что у attribute_spec.py: текст группы,
таблицы, на которые она ссылается, и общие положения. Результат — attribute_spec_free.json.

    venv2/Scripts/python -m graphs.structured_documents.attribute_spec_free --doc <папка документа>
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from .attribute_spec import BATCH, _cited_tables
from .modules.config_structured_documents_graph import Settings
from .rule_cards import GLOBAL_CHARS, _global_text, _invoke_json

load_dotenv()
settings = Settings()
OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "output"


def compile_free(rule_groups: list[dict], llm, doc_dir: Path) -> dict:
    from graphs.rag_agent.modules.tables import load_tables
    global_text = _global_text(rule_groups)
    local = [g for g in rule_groups if g.get("type") != "global"
             and g.get("kind", "presentation") == "presentation"]
    tables = load_tables(doc_dir)
    texts = ["\n".join(g.get("rules", [])) + _cited_tables(g, rule_groups, tables) for g in local]

    def run(start: int) -> dict[int, dict]:
        payload = {"GLOBAL": global_text[:GLOBAL_CHARS],
                   "groups": [{"id": start + i, "attribute": g.get("attribute", ""), "rules": texts[start + i]}
                              for i, g in enumerate(local[start:start + BATCH])]}
        for attempt in range(3):
            try:
                parsed = _invoke_json(llm, settings.FREE_SPEC_PROMPT, payload)
                return {s["id"]: s.get("spec") for s in parsed.get("specs", [])
                        if isinstance(s, dict) and isinstance(s.get("id"), int)}
            except Exception as e:
                print(f"  пачка {start}: {type(e).__name__}, повтор")
                time.sleep(5 * (attempt + 1))
        return {}

    with ThreadPoolExecutor(max_workers=5) as pool:
        by_id = {k: v for part in pool.map(run, range(0, len(local), BATCH)) for k, v in part.items()}
    return {"specs": [{"attribute": g.get("attribute", ""), "aliases": list(g.get("aliases") or []),
                       "rules": texts[i], "free": by_id.get(i) or {}} for i, g in enumerate(local)]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", required=True)
    doc = parser.parse_args().doc
    llm = ChatOpenAI(model=settings.llmodel, max_tokens=16000, temperature=0, timeout=300,
                     model_kwargs={"response_format": {"type": "json_object"}})
    path = OUTPUT_DIR / doc / "rule_groups.json"
    specs = compile_free(json.loads(path.read_text(encoding="utf-8")), llm, path.parent)
    (path.parent / "attribute_spec_free.json").write_text(json.dumps(specs, ensure_ascii=False, indent=1), encoding="utf-8")
    keys = {}
    for s in specs["specs"]:
        for k in s["free"]:
            keys[k] = keys.get(k, 0) + 1
    print(doc, f"описаний {sum(1 for s in specs['specs'] if s['free'])}/{len(specs['specs'])}; поля, которые выбрала модель:",
          dict(sorted(keys.items(), key=lambda x: -x[1])[:15]))


if __name__ == "__main__":
    main()
