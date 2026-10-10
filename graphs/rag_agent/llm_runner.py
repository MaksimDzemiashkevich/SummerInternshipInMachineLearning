# -*- coding: utf-8 -*-
"""Эксперименты B и C: правила применяет только модель, без кода пересчёта и форматирования.

Поиск правил тот же, что у SpecRunner (режим A): подпись строки → группа документа (модель с учётом
изделия, раздела его класса и значения), текст группы с таблицами, на которые она ссылается,
общие положения, корни обозначений, раздел класса изделия. Дальше значение пишет модель:
- «rules» (B): по тексту правил;
- «free» (C): по тексту правил и описанию атрибута, которое модель составила сама, выбрав поля
  (graphs/structured_documents/attribute_spec_free.py).
Уверенность — как в A для суждений модели: второй, слепой вызов с другим вопросом; совпало —
«средняя», разошлось — жёлтым с обоими вариантами; документ велит передать на проверку — жёлтым.

    from graphs.rag_agent.llm_runner import make_runner
    runner = make_runner(doc, model)   # режим из SPEC_MODE: code (A) | rules (B) | free (C)
"""
from __future__ import annotations

import json
import os

from .spec_runner import (BLIND_TEXT_PROMPT, OUTPUT_DIR, TEXT_PROMPT, SpecRunner, _ctx, _same_text)


class LLMRunner(SpecRunner):
    def __init__(self, doc: str, model, mode: str = "rules"):
        super().__init__(doc, model)
        self.mode = mode
        self.free: dict[str, dict] = {}
        if mode == "free":
            data = json.loads((OUTPUT_DIR / doc / "attribute_spec_free.json").read_text(encoding="utf-8"))
            self.free = {s["attribute"]: s["free"] for s in data["specs"]}

    def run(self, attribute: str, value: str, context: str = "") -> tuple[str, bool, str]:
        value = str(value).strip()
        spec = self.spec_for(attribute, context, value)
        parts = []
        if spec is None:
            parts.append("The document has no rules for this attribute; only its general rules apply.")
        else:
            parts.append(f"Rules of the document for this attribute:\n{spec['rules']}")
            if self.mode == "free" and self.free.get(spec["attribute"]):
                parts.append("Description of this attribute's rules, prepared from the document:\n"
                             + json.dumps(self.free[spec["attribute"]], ensure_ascii=False, indent=1))
        row_class = self._class_for(context)
        if row_class:
            parts.append(f"The document's section for this kind of item ({row_class}):\n{self.classes[row_class][:3000]}")
        roots = self._roots_for(attribute, spec)
        if roots:
            parts.append("Designation roots defined by the document (keyword -> root; the number of the record is "
                         "written right after the root, or before it where marked; where the rules spell a root "
                         "differently, this table wins):\n" + roots)
        question = (f"{_ctx(context)}Attribute: {attribute}\nRecorded value: {value!r}\n\n" + "\n\n".join(parts)
                    + f"\n\nGeneral rules of the document:\n{self.global_rules}")
        first = self._ask_text(TEXT_PROMPT, question)
        if first is None:
            return value, True, "низкая: ответ модели не разобран"
        written, flag = first
        if flag:
            return written, True, "документ велит оставить и передать на проверку"
        second = self._ask_text(BLIND_TEXT_PROMPT, question)
        if second is not None and _same_text(second[0], written):
            return written, False, "средняя: модель, сверено вслепую"
        return written, True, f"разошлись: «{written}» / «{second[0] if second else '—'}» — на проверку"


def make_runner(doc: str, model, mode: str | None = None):
    """A — «code» (спецификация + код), B — «rules» (только модель), C — «free» (модель + своё описание)."""
    mode = (mode or os.getenv("SPEC_MODE", "code")).strip().lower()
    return SpecRunner(doc, model) if mode == "code" else LLMRunner(doc, model, mode)
