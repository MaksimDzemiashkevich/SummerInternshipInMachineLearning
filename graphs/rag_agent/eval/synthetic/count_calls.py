# -*- coding: utf-8 -*-
"""Считает реальные вызовы модели и токены при нормализации одной таблицы."""
import sys
import threading
import time
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # корень репозитория
import openai.resources.chat.completions as cc
import openai.resources.embeddings as emb

lock = threading.Lock()
stat = Counter()
orig_create = cc.Completions.create


def counted_create(self, *args, **kwargs):
    response = orig_create(self, *args, **kwargs)
    body = response
    if hasattr(response, "parse"):  # with_raw_response
        body = response.parse()
    usage = getattr(body, "usage", None)
    with lock:
        stat["calls"] += 1
        if usage:
            stat["prompt"] += usage.prompt_tokens
            stat["completion"] += usage.completion_tokens
            details = getattr(usage, "prompt_tokens_details", None)
            stat["cached"] += getattr(details, "cached_tokens", 0) or 0
    return response


cc.Completions.create = counted_create
orig_emb = emb.Embeddings.create


def counted_emb(self, *args, **kwargs):
    with lock:
        stat["emb_calls"] += 1
    return orig_emb(self, *args, **kwargs)


emb.Embeddings.create = counted_emb

import logging
logging.disable(logging.INFO)
import pandas as pd
from graphs.rag_agent import api

name, xlsx = sys.argv[1], sys.argv[2]
cfg = api.get_app_config()
stat.clear()  # загрузка стандартов не в счёт
rs = next(r for r in cfg.rule_sets if r.name == name)
frame = api.prepare_frame(pd.read_excel(xlsx))
started = time.time()
out = api.run_single_rule_set(frame, cfg.model, rs)
elapsed = time.time() - started
rows = len(frame)
det = int(out["Deterministic"].sum())
items = frame["Item"].nunique() if "Item" in frame else 0
print(f"строк {rows}, изделий {items}, решено без LLM {det}, время {elapsed:.0f} с")
print(f"вызовов LLM: {stat['calls']} (≈ {stat['calls'] / rows:.1f} на строку); эмбеддингов: {stat['emb_calls']}")
print(f"токены: вход {stat['prompt']} (из них из кэша {stat['cached']}), выход {stat['completion']}")
print(f"на строку: вход ≈ {stat['prompt'] / rows:.0f}, выход ≈ {stat['completion'] / rows:.0f}")
