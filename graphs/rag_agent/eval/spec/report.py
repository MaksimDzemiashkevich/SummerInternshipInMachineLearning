# -*- coding: utf-8 -*-
"""Отчёт по прогону прототипа: четыре исхода строки, а не один процент.

Промышленные системы меряют себя не долей верных ответов, а долей МОЛЧАЛИВЫХ ошибок (неверно и
без жёлтой пометки) при разумной доле строк, отданных человеку. Поэтому каждая строка попадает в
один из исходов:
  верно            — значение верное, без пометки;
  ложная тревога   — значение верное, но помечено (лишняя работа ревьюеру);
  ошибка помечена  — значение неверное, но помечено (ревьюер увидит);
  молча            — значение неверное и без пометки (главная метрика, её доводим до нуля).
Отдельно — «пропущенная пометка»: значение верное, но документ велел передать его на проверку.
В конце — строка SUMMARY в JSON для сводки по всем наборам.
"""
from __future__ import annotations

import json
from collections import Counter


def report(name: str, rows: list[dict], seconds: float) -> None:
    """rows: [{attribute, value, actual, flag, note, good, expected, must_flag}]"""
    outcome = Counter()
    lines, missed = [], []
    for i, r in enumerate(rows):
        if r["good"] and (not r["flag"] or r.get("must_flag")):
            outcome["верно"] += 1  # пометка там, где её велит документ, — часть верного ответа
        elif r["good"]:
            outcome["ложная тревога"] += 1
        elif r["flag"]:
            outcome["ошибка помечена"] += 1
        else:
            outcome["молча"] += 1
        if r["good"] and r.get("must_flag") and not r["flag"]:
            missed.append(f"  ROW {i} {r['attribute']!r}/{r['value']!r}: верно, но документ велит пометить")
        if not r["good"] or (r["flag"] and not r.get("must_flag")):
            kind = "МОЛЧА" if not r["good"] and not r["flag"] else ("помечено" if not r["good"] else "ложная тревога")
            lines.append(f"  ROW {i} [{kind}] {r['attribute']!r}/{r['value']!r}\n      exp: {r['expected']}\n"
                         f"      act: {r['actual']!r}  ({r['note']})")
    tiers = Counter(str(r["note"]).split(":")[0].split(";")[-1].strip() for r in rows)
    total = len(rows)
    print(f"{name}: верно {outcome['верно']}/{total} | ложные тревоги {outcome['ложная тревога']} | "
          f"ошибки с пометкой {outcome['ошибка помечена']} | МОЛЧА {outcome['молча']} | "
          f"пропущенные пометки {len(missed)} | {seconds:.0f}s")
    print("  уровни:", dict(tiers.most_common()))
    if lines:
        print("\n".join(lines))
    if missed:
        print("\n".join(missed))
    print("SUMMARY " + json.dumps({"set": name, "total": total, **outcome, "missed_flag": len(missed)},
                                  ensure_ascii=False))
