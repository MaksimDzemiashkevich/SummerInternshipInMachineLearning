# -*- coding: utf-8 -*-
"""Роли перечней в тексте правил: закрытый список значений, синонимы или что-то другое.

Раньше рантайм сам искал в строках правил таблицы «описание → код» регулярками
(`_extract_table_pairs`) и угадывал роль колонки по виду значения. Угадывал плохо: таблица
коэффициентов пересчёта («GPM (US): 'x 0.2271' → M3/H») и строки тестовых примеров
(«4.5 in → R-SB-02») становились закрытыми реестрами, и любая величина такого атрибута
получала «pending registration»; таблица синонимов с колонкой форм не в конце не узнавалась.
Здесь модель один раз при загрузке документа читает каждую группу правил и говорит, какие в
ней перечни и что они значат. Код только проверяет, что обе стороны каждой пары буквально
стоят в тексте группы, — придумать соответствие модель не может. Результат — table_roles.json
рядом с rule_groups.json: {"lists": [{"group", "global", "title", "any_attribute", "role",
"pairs"}]}. Пары есть только у register (наименование → код) и synonyms (форма → основная запись).

Запуск (из корня репозитория) — по готовым rule_groups.json, без перегонки PDF:

    venv2/Scripts/python -m graphs.structured_documents.table_roles
    venv2/Scripts/python -m graphs.structured_documents.table_roles --doc OGS-STD-001_Standardization_Rules
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from .modules.config_structured_documents_graph import Settings
from .rule_cards import GLOBAL_CHARS, _global_text, _invoke_json, _norm
from graphs.paths import DOCUMENTS_OUTPUT as OUTPUT_DIR

load_dotenv()

settings = Settings()
ROLES = ("register", "synonyms", "conversion", "examples", "other")
# Текста групп в одном вызове: больше — модель начинает пропускать перечни в конце.
BATCH_CHARS = 10000
WORKERS = 5
ATTEMPTS = 3
# зависший запрос иначе держит шаг бесконечно, и повтор не наступает
REQUEST_TIMEOUT = 300
# Сколько раз классифицировать документ: роль пары решает большинство (classify_lists_voted).
VOTES = 3


def _chunks(rule_groups: list[dict]) -> list[dict]:
    """Части текста групп не длиннее BATCH_CHARS: длинная группа (общие положения) режется по
    строкам правил, перечень при этом может оказаться в двух частях — пары берутся из обеих."""
    parts: list[dict] = []
    for group in rule_groups:
        lines, size = [], 0
        for rule in group.get("rules", []):
            rule = str(rule)
            if lines and size + len(rule) > BATCH_CHARS:
                parts.append({"group": group, "text": "\n".join(lines)})
                lines, size = [], 0
            lines.append(rule)
            size += len(rule)
        if lines:
            parts.append({"group": group, "text": "\n".join(lines)})
    return parts


def _batches(parts: list[dict]) -> list[list[dict]]:
    batches: list[list[dict]] = [[]]
    size = 0
    for part in parts:
        if batches[-1] and size + len(part["text"]) > BATCH_CHARS:
            batches.append([])
            size = 0
        batches[-1].append(part)
        size += len(part["text"])
    return [b for b in batches if b]


def _grounded(pairs, text: str) -> list[dict]:
    """Только пары, у которых и форма, и запись буквально стоят в тексте группы."""
    haystack = _norm(text)
    kept = []
    for pair in pairs or []:
        if not isinstance(pair, dict):
            continue
        form, value = str(pair.get("form", "")).strip(), str(pair.get("value", "")).strip()
        if not form or not value or _norm(form) == _norm(value):
            continue
        if _norm(form) in haystack and _norm(value) in haystack:
            kept.append({"form": form, "value": value})
    return kept


def classify_lists(rule_groups: list[dict], llm) -> dict:
    """{"lists": [...]} — перечни всех групп документа с их ролью и проверенными парами."""
    parts = _chunks([g for g in rule_groups if g.get("kind", "presentation") != "class"])
    global_text = _global_text(rule_groups)

    def run(batch: list[dict]) -> list[dict]:
        ids = {id(p): i for i, p in enumerate(batch)}
        payload = {"GLOBAL": global_text[:GLOBAL_CHARS], "groups": [{"id": ids[id(p)], "attribute": p["group"].get("attribute", ""),
                               "rules": p["text"]} for p in batch]}
        # сбой сети не должен оставить документ без реестров: повторяем, а не дождавшись
        # ответа — падаем, и прежний table_roles.json остаётся на месте
        for attempt in range(1, ATTEMPTS + 1):
            try:
                parsed = _invoke_json(llm, settings.TABLE_ROLES_PROMPT, payload)
                break
            except Exception as e:
                print(f"TABLE ROLES ERROR (попытка {attempt}/{ATTEMPTS}):", type(e).__name__, str(e)[:200])
                if attempt == ATTEMPTS:
                    raise
                time.sleep(5 * attempt)
        found = []
        for entry in parsed.get("groups", []):
            if not isinstance(entry, dict) or not isinstance(entry.get("id"), int) \
                    or not 0 <= entry["id"] < len(batch):
                continue
            part = batch[entry["id"]]
            for item in entry.get("lists") or []:
                if not isinstance(item, dict) or item.get("role") not in ROLES:
                    continue
                role = item["role"]
                # таблица кодов лишь для части значений («standard → SCH STD», а «SCH 10S» —
                # по формату правила) реестром быть не может: всё, чего в ней нет, стало бы
                # «pending registration». Её пары — подстановки, как у синонимов.
                if role == "register" and item.get("whole_attribute") is not True:
                    role = "synonyms"
                pairs = (_grounded(item.get("pairs"), part["text"])
                         if role in ("register", "synonyms") else [])
                group_name = part["group"].get("attribute", "")
                found.append({"group": group_name,
                              # чьи значения перечислены: группа-хозяин текста — не всегда он
                              "attribute": str(item.get("attribute") or "").strip() or group_name,
                              "global": part["group"].get("type") == "global",
                              "title": str(item.get("title") or ""),
                              "any_attribute": item.get("any_attribute") is True,
                              "role": role, "pairs": pairs})
        return found

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(run, _batches(parts)))
    return {"lists": [item for part in results for item in part]}


def classify_lists_voted(rule_groups: list[dict], llm, runs: int = VOTES) -> dict:
    """classify_lists несколько раз; роль каждой пары — по большинству прогонов.

    Граница «реестр кодов или синонимы» у модели неустойчива: от прогона к прогону одна и та же
    таблица кодов то становилась реестром, то нет (CWX, QRS, OGX теряли реестры, которые раньше
    находились всегда). Пара, которую нашло меньшинство прогонов, отбрасывается: найденное один
    раз из трёх — скорее случайность, чем таблица документа. Перечни без пар (пересчёт, примеры,
    прочее) берутся из первого прогона: рантайм их не использует.
    """
    rounds = [classify_lists(rule_groups, llm) for _ in range(runs)]
    need = runs // 2 + 1
    votes: dict[tuple, Counter] = {}
    first_seen: dict[tuple, dict] = {}
    for round_ in rounds:
        seen_here: set[tuple] = set()
        for item in round_["lists"]:
            for pair in item["pairs"]:
                key = (item["attribute"].strip().lower(), _norm(pair["form"]), _norm(pair["value"]))
                if key in seen_here:
                    continue  # один прогон голосует за пару один раз
                seen_here.add(key)
                votes.setdefault(key, Counter())[(item["role"], item["any_attribute"])] += 1
                first_seen.setdefault(key, {**item, "pair": pair})
    merged: dict[tuple, dict] = {}
    for key, counter in votes.items():
        if sum(counter.values()) < need:
            continue
        # при ничьей — синонимы: ошибочный реестр делает «pending registration» из всего, чего
        # в нём нет, а ошибочные синонимы лишь отдают такую строку модели
        role, any_attribute = max(counter, key=lambda k: (counter[k], k[0] == "synonyms"))
        item = first_seen[key]
        list_key = (item["attribute"], item["title"], role, any_attribute)
        merged.setdefault(list_key, {"group": item["group"], "attribute": item["attribute"],
                                     "global": item["global"], "title": item["title"],
                                     "any_attribute": any_attribute, "role": role, "pairs": []})
        merged[list_key]["pairs"].append(item["pair"])
    without_pairs = [i for i in rounds[0]["lists"] if not i["pairs"]]
    return {"lists": list(merged.values()) + without_pairs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", default="", help="имя папки документа (по умолчанию — все)")
    args = parser.parse_args()
    llm = ChatOpenAI(model=settings.llmodel, max_tokens=16000, temperature=0, timeout=REQUEST_TIMEOUT,
                     model_kwargs={"response_format": {"type": "json_object"}})
    for path in sorted(OUTPUT_DIR.glob("*/rule_groups.json")):
        if args.doc and path.parent.name != args.doc:
            continue
        try:
            roles = classify_lists_voted(json.loads(path.read_text(encoding="utf-8")), llm)
        except Exception as e:
            print(f"{path.parent.name}: НЕ ОБНОВЛЁН ({type(e).__name__}) — прежний файл оставлен")
            continue
        (path.parent / "table_roles.json").write_text(
            json.dumps(roles, ensure_ascii=False, indent=1), encoding="utf-8")
        summary = []
        for role in ROLES:
            items = [i for i in roles["lists"] if i["role"] == role]
            if items:
                pairs = sum(len(i["pairs"]) for i in items)
                summary.append(f"{role} {len(items)}" + (f" ({pairs} пар)" if pairs else ""))
        print(f"{path.parent.name}: " + (", ".join(summary) or "перечней нет"))


if __name__ == "__main__":
    main()
