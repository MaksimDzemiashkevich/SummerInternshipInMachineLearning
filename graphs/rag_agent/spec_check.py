# -*- coding: utf-8 -*-
"""Прототип: проверка спецификаций атрибутов на примерах самого документа и их исправление.

Спецификацию (attribute_spec.json) составила модель, и её ошибки — лишние знаки, проза в поле
единицы, пропущенная принятая форма — раньше были видны только на замере. Здесь спецификация
исполняется (spec_runner) на парах «было → стало» из документа (rule_cards.json, обе стороны
сверены с текстом), несошедшиеся пары отдаются модели вместе со спецификацией и правилами,
и исправленная версия принимается, только если сходится больше примеров (как в TabClean:
лучшая проверенная версия не может стать хуже).

    venv2/Scripts/python -m graphs.rag_agent.spec_check --doc <папка документа>
"""
from __future__ import annotations

import argparse
import copy
import json
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from graphs.structured_documents.attribute_spec import _ground
from graphs.structured_documents.modules.config_structured_documents_graph import Settings
from graphs.structured_documents.rule_cards import _norm

from .spec_runner import OUTPUT_DIR, SpecRunner

load_dotenv()
settings = Settings()
ROUNDS = 2


def _score(runner: SpecRunner, spec: dict, examples: list[dict]) -> list[dict]:
    """Несошедшиеся примеры спецификации: [{from, expected, got}]."""
    runner._spec_cache.clear()
    original = runner.specs
    runner.specs = [spec]
    try:
        fails = []
        for e in examples:
            try:
                got = runner.run(spec["attribute"], e["from"])[0]
            except Exception as exc:  # пример не должен ронять проверку
                got = f"<error {type(exc).__name__}>"
            if got.strip() != e["to"].strip():
                fails.append({"from": e["from"], "expected": e["to"], "got": got})
        return fails
    finally:
        runner.specs = original


def check_doc(doc: str, model: ChatOpenAI) -> None:
    folder = OUTPUT_DIR / doc
    cards = json.loads((folder / "rule_cards.json").read_text(encoding="utf-8")).get("groups", {})
    data = json.loads((folder / "attribute_spec.json").read_text(encoding="utf-8"))
    runner = SpecRunner(doc, model)
    repair_llm = ChatOpenAI(model=settings.llmodel, temperature=0, timeout=300,
                            model_kwargs={"response_format": {"type": "json_object"}})

    def work(index: int):
        spec = data["specs"][index]
        examples = (cards.get(spec["attribute"]) or {}).get("examples") or []
        if not examples:
            return index, spec, 0, 0, len(examples)
        local = SpecRunner.__new__(SpecRunner)
        local.__dict__.update({**runner.__dict__, "_spec_cache": {}, "_lock": runner._lock})
        best, fails = spec, _score(local, spec, examples)
        before = len(fails)
        for _ in range(ROUNDS):
            if not fails:
                break
            payload = {"spec": {k: v for k, v in best.items() if k not in ("rules", "attribute", "aliases")},
                       "rules": best["rules"], "general_rules": runner.global_rules[:4000], "failed_examples": fails}
            try:
                fixed = json.loads(str(repair_llm.invoke([
                    {"role": "system", "content": settings.SPEC_REPAIR_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]).content))
            except Exception:
                break
            candidate = {**copy.deepcopy(best), **{k: v for k, v in fixed.items() if k not in ("rules", "attribute", "aliases")}}
            text = _norm(best["rules"] + "\n" + runner.global_rules)
            _ground(candidate, text)
            for variant in candidate.get("variants") or []:
                _ground(variant, text)
            candidate_fails = _score(local, candidate, examples)
            if len(candidate_fails) < len(fails):
                best, fails = candidate, candidate_fails
            else:
                break
        return index, best, before, len(fails), len(examples)

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(work, range(len(data["specs"]))))
    total = sum(r[4] for r in results)
    before = sum(r[2] for r in results)
    after = sum(r[3] for r in results)
    for index, spec, *_ in results:
        data["specs"][index] = spec
    (folder / "attribute_spec.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{doc}: примеров {total}, не сходилось {before} → {after}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", required=True)
    args = parser.parse_args()
    check_doc(args.doc, ChatOpenAI(model=settings.llmodel, temperature=0, timeout=180))


if __name__ == "__main__":
    main()
