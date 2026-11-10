# -*- coding: utf-8 -*-
"""Прототип на синтетическом наборе с эталоном eval/synthetic/expected/*.json.

    venv2/Scripts/python graphs/rag_agent/eval/spec/score_expected.py <папка документа> <input.xlsx> <expected.json>
"""
import json
import re
import os
import sys
import time

sys.path.insert(0, '.')
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
from graphs.rag_agent.eval.spec.report import report
from graphs.rag_agent.llm_runner import make_runner
from graphs.structured_documents.modules.config_structured_documents_graph import Settings

doc, xlsx, exp = sys.argv[1:4]
runner = make_runner(doc, ChatOpenAI(model=Settings().llmodel, temperature=0, timeout=180))
frame = pd.read_excel(xlsx).fillna("")
expected = json.load(open(exp, encoding='utf-8'))


def one(row):
    try:
        return runner.run(row['Attribute'], str(row['Value']), str(row.get('Item', '')))
    except Exception as e:  # строка не должна ронять замер
        return str(row['Value']), True, f"ОШИБКА {type(e).__name__}: {e}"[:150]


started = time.time()
with ThreadPoolExecutor(6) as pool:
    out = list(pool.map(one, [r for _, r in frame.iterrows()]))
rows = []
for i, ((actual, flag, note), e) in enumerate(zip(out, expected)):
    options = e['expected'] if isinstance(e['expected'], list) else [e['expected']]
    good = any((re.search(o['re'], actual) if isinstance(o, dict) else actual == o) for o in options)
    rows.append({"attribute": frame.Attribute[i], "value": frame.Value[i], "actual": actual, "flag": flag,
                 "note": note, "good": good, "expected": options, "must_flag": e.get('review')})
report(f"{doc} [{os.getenv('SPEC_MODE', 'code')}]", rows, time.time() - started)
