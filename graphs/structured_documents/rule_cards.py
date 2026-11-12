# -*- coding: utf-8 -*-
"""Карточки правил: документ толкуется один раз при загрузке, а не заново на каждой строке.

Раньше модель на КАЖДОЙ строке получала сырую прозу документа и решала, что автор имел в виду,
и решала по-разному: «pipe is Seamless (‘SMLS’ and ‘S/L’ accepted)» в одном прогоне давало
«Seamless», в другом — «SMLS». Здесь модель один раз пересказывает каждую группу правил в
однозначную карточку: форма записи, единица и знаки, допустимые значения, принятые формы →
основная, шаги, исключения, неясности. Принятые формы проверяются кодом: и форма, и основная
запись обязаны буквально стоять в тексте документа, иначе пара выбрасывается — модель не может
выдумать соответствие. Результат — rule_cards.json рядом с rule_groups.json:
{"global": {...}, "groups": {"<имя группы>": {...}}}.

Запуск (из корня репозитория) — по готовым rule_groups.json, без перегонки PDF:

    venv2/Scripts/python -m graphs.structured_documents.rule_cards
    venv2/Scripts/python -m graphs.structured_documents.rule_cards --doc MNT-DN-014_equipment_data_working_notes
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .modules.config_structured_documents_graph import Settings
from graphs.paths import DOCUMENTS_OUTPUT as OUTPUT_DIR

load_dotenv()

settings = Settings()
# Сколько групп пересказывается за один вызов: общие положения идут в каждый вызов
# контекстом, поэтому группы собираются пачками, а не по одной.
BATCH = 8
WORKERS = 5
# Общие положения в контексте пачки: у 82-страничного документа их 20 тыс. символов.
# Общие положения целиком: у 82-страничного документа их ~26 тыс. символов, и обрезка на 12 тыс.
# отрезала правило «нули в конце дробной части убрать» (EMD 7.3) — ни один шаг его не видел.
GLOBAL_CHARS = 60000


def _norm(text: str) -> str:
    """Для проверки «стоит ли это в тексте»: регистр, кавычки и пробелы не важны."""
    text = re.sub(r"[‘’“”'\"]", "", str(text).lower())
    return re.sub(r"\s+", " ", text).strip()


def _grounded_pairs(pairs, source: str) -> list[dict]:
    """Только пары, у которых и форма, и основная запись буквально есть в тексте документа."""
    haystack = _norm(source)
    kept = []
    for pair in pairs or []:
        if not isinstance(pair, dict):
            continue
        form, written = str(pair.get("form", "")).strip(), str(pair.get("written", "")).strip()
        if not form or not written or form == written:
            continue
        if _norm(form) in haystack and _norm(written) in haystack:
            kept.append({"form": form, "written": written})
    return kept


def _grounded_examples(pairs, source: str) -> list[dict]:
    """Примеры «было → стало»: только те, у которых обе стороны буквально стоят в тексте."""
    haystack = _norm(source)
    kept = []
    for pair in pairs or []:
        if not isinstance(pair, dict):
            continue
        before, after = str(pair.get("from", "")).strip(), str(pair.get("to", "")).strip()
        if before and after and _norm(before) in haystack and _norm(after) in haystack:
            kept.append({"from": before, "to": after})
    return kept


def _spacing(value) -> str:
    """Пробел между числом и единицей по карточке: «space», «none» или «» (документ молчит)."""
    return value if value in ("space", "none") else ""


def _invoke_json(llm, prompt: str, payload: dict) -> dict:
    """Вызов модели с ответом JSON."""
    response = llm.invoke([SystemMessage(content=prompt),
                           HumanMessage(content=json.dumps(payload, ensure_ascii=False))])
    return json.loads(str(response.content))


def _global_text(rule_groups: list[dict]) -> str:
    """Общие положения документа одним текстом."""
    return "\n".join(r for g in rule_groups if g.get("type") == "global" for r in g.get("rules", []))


def compile_global_card(rule_groups: list[dict], llm) -> dict:
    """Карточка общих положений документа."""
    global_text = _global_text(rule_groups)
    try:
        global_card = _invoke_json(llm, settings.GLOBAL_CARD_PROMPT, {"GLOBAL": global_text})
    except Exception as e:
        print("GLOBAL CARD ERROR:", type(e).__name__, str(e)[:200])
        global_card = {}
    global_card["accepted"] = _grounded_pairs(global_card.get("accepted"), global_text)
    global_card["examples"] = _grounded_examples(global_card.get("examples"), global_text)
    global_card["unit_spacing"] = _spacing(global_card.get("unit_spacing"))
    # о нулях в конце дробной части общие положения должны говорить сами — иначе поле выдумано
    if global_card.get("trailing_zeros") not in ("remove", "keep") or not re.search(
            r"\bzeros?\b", global_text, re.I):
        global_card["trailing_zeros"] = ""
    return global_card


def compile_cards(rule_groups: list[dict], llm) -> dict:
    """{"global": карточка общих положений, "groups": {имя группы: карточка}}."""
    global_text = _global_text(rule_groups)
    local = [g for g in rule_groups
             if g.get("type") != "global" and g.get("kind", "presentation") == "presentation"]
    global_card = compile_global_card(rule_groups, llm)

    def batch_cards(start: int) -> dict[int, dict]:
        chunk = local[start:start + BATCH]
        payload = {"GLOBAL": global_text[:GLOBAL_CHARS],
                   "groups": [{"id": start + i, "attribute": g.get("attribute", ""),
                               "rules": "\n".join(g.get("rules", []))}
                              for i, g in enumerate(chunk)]}
        try:
            parsed = _invoke_json(llm, settings.RULE_CARD_PROMPT, payload)
        except Exception as e:
            print(f"CARD ERROR (группы {start}–{start + len(chunk) - 1}):", type(e).__name__, str(e)[:200])
            return {}
        return {c["id"]: c for c in parsed.get("cards", [])
                if isinstance(c, dict) and isinstance(c.get("id"), int)}

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(batch_cards, range(0, len(local), BATCH)))
    by_id = {k: v for part in results for k, v in part.items()}

    cards: dict[str, dict] = {}
    for i, group in enumerate(local):
        card = by_id.get(i)
        if not card:
            continue
        body = "\n".join(group.get("rules", []))
        card = {k: v for k, v in card.items() if k != "id"}
        # форма должна стоять в тексте группы или в общих положениях — иначе выдумана
        card["accepted"] = _grounded_pairs(card.get("accepted"), body + "\n" + global_text)
        card["examples"] = _grounded_examples(card.get("examples"), body)
        card["unit_spacing"] = _spacing(card.get("unit_spacing"))
        cards[group["attribute"]] = card
    missing = [g["attribute"] for i, g in enumerate(local) if i not in by_id]
    if missing:
        print(f"  без карточки осталось групп: {len(missing)} — {missing[:8]}")
    return {"global": global_card, "groups": cards}


def main() -> None:
    """CLI: собирает rule_cards.json для документа --doc или для всех."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", default="", help="имя папки документа (по умолчанию — все)")
    parser.add_argument("--global-only", action="store_true",
                        help="пересобрать только общую карточку, карточки групп оставить как есть")
    args = parser.parse_args()

    llm = ChatOpenAI(model=settings.llmodel, max_tokens=16000, temperature=0,
                     model_kwargs={"response_format": {"type": "json_object"}})

    for path in sorted(OUTPUT_DIR.glob("*/rule_groups.json")):
        if args.doc and path.parent.name != args.doc:
            continue
        rule_groups = json.loads(path.read_text(encoding="utf-8"))
        cards_path = path.parent / "rule_cards.json"
        if args.global_only and cards_path.exists():
            cards = json.loads(cards_path.read_text(encoding="utf-8"))
            cards["global"] = compile_global_card(rule_groups, llm)
        else:
            cards = compile_cards(rule_groups, llm)
        (path.parent / "rule_cards.json").write_text(
            json.dumps(cards, ensure_ascii=False, indent=1), encoding="utf-8")
        accepted = sum(len(c.get("accepted", [])) for c in cards["groups"].values())
        examples = sum(len(c.get("examples", [])) for c in cards["groups"].values())
        print(f"{path.parent.name}: карточек {len(cards['groups'])}, принятых форм {accepted}, "
              f"примеров {examples}, нули: {cards['global'].get('trailing_zeros') or '—'}, "
              f"общих шагов {len(cards['global'].get('steps', []))}")


if __name__ == "__main__":
    main()
