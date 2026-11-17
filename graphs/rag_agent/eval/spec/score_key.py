# -*- coding: utf-8 -*-
"""Замер на таблице с правильными ответами (ключ — отдельный файл, колонка ожидаемых значений).

    venv2/Scripts/python graphs/rag_agent/eval/spec/score_key.py <папка документа> <input.xlsx> <key.xlsx> "<колонка ключа>"
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
from graphs.rag_agent.eval.spec.report import report
from graphs.rag_agent.spec_runner import SpecRunner
from graphs.structured_documents.modules.config_structured_documents_graph import Settings

doc, src, keyf, col = sys.argv[1:5]
runner = SpecRunner(doc, ChatOpenAI(model=Settings().llmodel, temperature=0, timeout=180))
frame = pd.read_excel(src).fillna("")
key = pd.read_excel(keyf)
text = lambda x: '' if pd.isna(x) else str(x).strip()


def one(row):
    try:
        return runner.run(row['Attribute'], str(row['Value']), str(row.get('Item', '')))
    except Exception as e:  # строка не должна ронять замер
        return str(row['Value']), True, f"ОШИБКА {type(e).__name__}: {e}"[:150]


started = time.time()
with ThreadPoolExecutor(6) as pool:
    out = list(pool.map(one, [r for _, r in frame.iterrows()]))
rows = []
for i, (actual, flag, note) in enumerate(out):
    expected = text(key[col][i])
    rows.append({"attribute": frame.Attribute[i], "value": frame.Value[i], "actual": actual, "flag": flag,
                 "note": note, "good": actual == expected, "expected": expected, "must_flag": None})
report(f"{doc} [spec]", rows, time.time() - started)
