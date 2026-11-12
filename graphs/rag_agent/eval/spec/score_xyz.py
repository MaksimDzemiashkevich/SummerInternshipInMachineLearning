# -*- coding: utf-8 -*-
"""Прототип на наборе run_eval (XYZ STD 100 / 205) с его эталоном и обязательными пометками.

    venv2/Scripts/python graphs/rag_agent/eval/spec/score_xyz.py
"""
import os
import sys
import time

sys.path.insert(0, '.')
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
from graphs.rag_agent.eval.expected import EXPECTED, REVIEW_FORBIDDEN, REVIEW_REQUIRED
from graphs.rag_agent.eval.run_eval import INPUT, S100, S205, matches, show
from graphs.rag_agent.eval.spec.report import report
from graphs.rag_agent.spec_runner import SpecRunner
from graphs.structured_documents.modules.config_structured_documents_graph import Settings

model = ChatOpenAI(model=Settings().llmodel, temperature=0, timeout=180)
runners = {n: SpecRunner(n, model) for n in (S100, S205)}
df = pd.read_excel(INPUT).fillna("")
jobs = [(i, n) for i in range(len(df)) for n in (S100, S205)]


def one(job):
    i, n = job
    try:
        return runners[n].run(df.loc[i, 'Attribute'], str(df.loc[i, 'Value']))
    except Exception as e:  # строка не должна ронять замер
        return str(df.loc[i, 'Value']), True, f"ОШИБКА {e}"[:120]


started = time.time()
with ThreadPoolExecutor(6) as pool:
    results = dict(zip(jobs, pool.map(one, jobs)))
rows = []
for i in range(len(df)):
    attribute, value, e100, e205 = EXPECTED[i]
    for n, e in ((S100, e100), (S205, e205)):
        if isinstance(e, dict) and e.get("skip"):
            continue  # спорная строка самих документов
        actual, flag, note = results[(i, n)]
        must = True if (i, n) in REVIEW_REQUIRED else (False if (i, n) in REVIEW_FORBIDDEN else None)
        rows.append({"attribute": f"[{n[8:11]}] {attribute}", "value": value, "actual": actual, "flag": flag,
                     "note": note, "good": matches(e, actual), "expected": show(e), "must_flag": must})
report(f"XYZ run_eval [spec]", rows, time.time() - started)
