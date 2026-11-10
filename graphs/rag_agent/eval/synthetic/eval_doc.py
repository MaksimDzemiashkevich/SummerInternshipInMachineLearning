# -*- coding: utf-8 -*-
"""Прогон одного документа по датащиту со сверкой с эталоном.

    venv2/Scripts/python eval_doc.py <префикс папки документа> <xlsx> <expected.json> [rows]
expected.json: список {"expected": str | [str,...], "review": bool | null} по строкам xlsx.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # корень репозитория

import pandas as pd

from graphs.rag_agent import api


def main():
    prefix, xlsx, expected_path = sys.argv[1:4]
    rows = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else None
    expected = json.load(open(expected_path, encoding="utf-8"))
    cfg = api.get_app_config()
    rule_set = next(r for r in cfg.rule_sets if r.name == prefix) if any(
        r.name == prefix for r in cfg.rule_sets) else next(r for r in cfg.rule_sets if r.name.startswith(prefix))
    frame = api.prepare_frame(pd.read_excel(xlsx))
    if rows:
        frame = frame.iloc[rows].reset_index(drop=True)
        expected = [expected[i] for i in rows]
    else:
        rows = list(range(len(frame)))
    started = time.time()
    out = api.run_single_rule_set(frame, cfg.model, rule_set)
    ok = 0
    bad, flags = [], []
    for pos, row in enumerate(rows):
        exp = expected[pos]["expected"]
        options = exp if isinstance(exp, list) else [exp]
        actual = str(out.iloc[pos]["Rules_Value"]).strip()
        flagged = bool(out.iloc[pos]["Ambiguous"])
        import re as _re
        if any((_re.search(o["re"], actual) if isinstance(o, dict) else actual == o) for o in options):
            ok += 1
        else:
            bad.append(f"  ROW {row:>3} {frame.loc[pos, 'Attribute']!r} / {frame.loc[pos, 'Value']!r}\n"
                       f"      expected: {' | '.join(o['re'] if isinstance(o, dict) else o for o in options)}\n      actual  : {actual!r}"
                       f"{'  [review]' if flagged else ''}\n      applied : {out.iloc[pos]['Applied'][:220]}")
        want = expected[pos].get("review")
        if want is True and not flagged:
            flags.append(f"  ROW {row:>3} не помечен, а должен: {actual!r}")
        if want is False and flagged and actual not in options:
            pass  # неверный и помеченный — уже в расхождениях
        elif want is False and flagged:
            flags.append(f"  ROW {row:>3} помечен зря: {actual!r}")
    print("=" * 78)
    print(f"{rule_set.name}\nACCURACY: {ok}/{len(rows)} = {ok / len(rows):.1%}  ({time.time() - started:.0f}s)")
    if bad:
        print("MISMATCHES:\n" + "\n".join(bad))
    if flags:
        print("REVIEW FLAGS:\n" + "\n".join(flags))


if __name__ == "__main__":
    main()
