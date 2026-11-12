# -*- coding: utf-8 -*-
"""Регрессионный прогон пайплайна нормализации со сверкой по эталону.

Запуск из корня репозитория:

    venv2/Scripts/python -m graphs.rag_agent.eval.run_eval
    venv2/Scripts/python -m graphs.rag_agent.eval.run_eval --rows 6,8,9
    NORMALIZER_SAMPLES=5 venv2/Scripts/python -m graphs.rag_agent.eval.run_eval

Печатает точность, список расхождений с эталоном и пары строк, которые обязаны были
дать побуквенно одинаковый результат. Результат кладётся в data/output/eval_<tag>.xlsx.
"""
from __future__ import annotations

import argparse
import re
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from graphs.paths import SERVICE_DATA, SERVICE_INPUT

from ..api import run_single_rule_set
from ..modules.config import build_app_config, Settings
from .expected import (EXPECTED, CONSISTENT_PAIRS, REVIEW_REQUIRED, REVIEW_FORBIDDEN,
                       S100, S205)

INPUT = SERVICE_INPUT / "test_attributes_api600_new_new.xlsx"
OUTPUT_DIR = SERVICE_DATA / "output"


def matches(expected, actual: str) -> bool:
    """Эталон — точная строка, регулярка {"re": ...} или список вариантов {"any": [...]}"""
    actual = str(actual).strip()
    if isinstance(expected, str):
        return actual == expected.strip()
    if isinstance(expected, dict):
        if "re" in expected:
            return re.search(expected["re"], actual) is not None
        if "any" in expected:
            return any(actual == e.strip() for e in expected["any"])
    return False


def show(expected) -> str:
    if isinstance(expected, str):
        return expected
    if "re" in expected:
        return f"/{expected['re']}/"
    return " | ".join(expected["any"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", default="", help="индексы строк через запятую (по умолчанию все)")
    parser.add_argument("--tag", default="run", help="суффикс имени выходного файла")
    args = parser.parse_args()

    data_frame = pd.read_excel(INPUT)
    indices = [int(x) for x in args.rows.split(",") if x.strip()] or list(range(len(data_frame)))
    subset = data_frame.iloc[indices].reset_index(drop=True)

    app_config = build_app_config(Settings())
    rule_sets = {rs.name: rs for rs in app_config.rule_sets}
    names = [S100, S205]

    # Строки внутри одного документа НЕ распараллеливаются: пост-проходы (сверка
    # эквивалентных строк, канонизация написания) работают на всей таблице сразу.
    started = time.time()
    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        outputs = list(pool.map(
            lambda name: run_single_rule_set(subset, app_config.model, rule_sets[name]), names))
    results = dict(zip(names, outputs))
    elapsed = time.time() - started

    ok = bad = 0
    mismatches: list[str] = []
    for position, row in enumerate(indices):
        attribute, value, expected_100, expected_205 = EXPECTED[row]
        for name, expected in ((S100, expected_100), (S205, expected_205)):
            if isinstance(expected, dict) and expected.get("skip"):
                continue  # спорная строка самих документов — в счёт не идёт
            actual = str(results[name].iloc[position]["Rules_Value"]).strip()
            if matches(expected, actual):
                ok += 1
            else:
                bad += 1
                mismatches.append(f"  ROW {row:>2} [{name[8:11]}] {attribute!r} / {value!r}\n"
                                  f"      expected: {show(expected)}\n"
                                  f"      actual  : {actual!r}")

    # Пометка на ручную проверку — часть результата: код, присуждённый не по точному
    # совпадению формулировки, обязан быть виден ревьюеру, а точное совпадение — нет.
    flags: list[str] = []
    for position, row in enumerate(indices):
        for name in names:
            flagged = bool(results[name].iloc[position].get("Ambiguous", False))
            if (row, name) in REVIEW_REQUIRED and not flagged:
                flags.append(f"  ROW {row:>2} [{name[8:11]}] не помечен, а должен быть: "
                             f"{results[name].iloc[position]['Rules_Value']!r}")
            if (row, name) in REVIEW_FORBIDDEN and flagged:
                flags.append(f"  ROW {row:>2} [{name[8:11]}] помечен зря: "
                             f"{results[name].iloc[position]['Rules_Value']!r}")

    inconsistent: list[str] = []
    for left, right in CONSISTENT_PAIRS:
        if left in indices and right in indices:
            for name in names:
                a = str(results[name].iloc[indices.index(left)]["Rules_Value"]).strip()
                b = str(results[name].iloc[indices.index(right)]["Rules_Value"]).strip()
                if a != b:
                    inconsistent.append(f"  ROWS {left}/{right} [{name[8:11]}] {a!r} != {b!r}")

    out_path = OUTPUT_DIR / f"eval_{args.tag}.xlsx"
    report = subset[["Attribute", "Value"]].copy()
    for name in names:
        report[name] = results[name]["Rules_Value"].to_list()
    report.to_excel(out_path, index=False)

    print("\n" + "=" * 78)
    print(f"ACCURACY: {ok}/{ok + bad} = {ok / max(ok + bad, 1):.1%}   ({elapsed:.0f}s)")
    if mismatches:
        print("\nMISMATCHES:\n" + "\n".join(mismatches))
    if flags:
        print("\nREVIEW FLAGS:\n" + "\n".join(flags))
    if inconsistent:
        print("\nINCONSISTENT PAIRS:\n" + "\n".join(inconsistent))
    print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
