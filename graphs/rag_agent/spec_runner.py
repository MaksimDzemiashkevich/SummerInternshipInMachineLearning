# -*- coding: utf-8 -*-
"""Прототип: исполнение спецификации атрибута (graphs/structured_documents/attribute_spec.py).

Модель выбирает, код пишет. Список: точное совпадение формы, иначе выбор модели, ограниченный
схемой — одно из значений списка или UNMATCHED (код вне списка выдумать нельзя). Величина:
модель разбирает число и единицу (pipeline/parsing.py), пересчёт по коэффициенту документа (или
pint), ряд, округление и запись по формату делает код. Формат, текст и всё, что код не взял на себя
(пустое значение, «нет в списке», диапазон): модель по инструкции спецификации и общим правилам
документа. Пометка ставится там, где документ велит «оставить и передать на проверку».

У каждого ответа — уровень уверенности по доказательствам (первое слово пояснения): «высокая» —
основание проверено кодом как факт (форма есть в таблице, цитата правила есть в тексте, пересчёт
сделал код); «средняя» — суждение модели, которое независимо повторил второй, слепой вызов;
«низкая» и «разошлись» — жёлтым, с предложенным значением. Смысл код не судит: прежний фильтр по
составу слов отклонял верные эквиваленты («300#» = «Class 300») и снят.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from langchain_openai import ChatOpenAI

from graphs.structured_documents.number_format import format_number, keep_digits, valid_locale, valid_number_pattern

from .modules.tables import form_key
from .modules.tools import _unit, convert_value, unit_mismatch
from .pipeline.common import _attribute_matches, _norm_attr, logger
from .pipeline.deterministic import _wording_matches
from .pipeline.parsing import parse

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "structured_documents" / "data" / "output"
GLOBAL_CHARS = 60000  # общие положения целиком: обрезка теряла правила из их середины

CHOICE_PROMPT = """You map one recorded value to the list of values a normalization document allows for this attribute. The rules of the document for this attribute are given.
Pick the list value the recorded value means under these rules: the same wording (word order, case, punctuation may differ), a form the rules accept, a value a rule maps it to, or an established equivalent notation of the very same value (an abbreviation, a class written with a unit name, the same designation written differently). A different, narrower, broader or branded product, or anything the list does not name, is "UNMATCHED": never pick the closest entry.
Say on what basis: "same wording"; "rule" (copy the rule sentence verbatim into rule_quote, otherwise rule_quote is ""); "equivalent" (an equivalent notation of the same value); "unsure" (you lean to this value but it may be a different one)."""

BLIND_CHOICE_PROMPT = """You check a data sheet value against a normalization document on your own, without seeing anyone else's answer. Read the rules, then say which listed value the recorded value is, or "UNMATCHED" if it is none of them (a different, narrower, broader or branded product is UNMATCHED — never the closest entry)."""

VARIANT_PROMPT = """A normalization document writes this attribute differently in different cases — for different kinds of items, or for different kinds of recorded values. Say which variant applies to this row: judge the item by what it is (its head noun, its kind), not by tag numbers; judge the value by what it is (its magnitude, the kind of quantity or reference it states). Answer "none of these" if none applies."""

CLASS_PROMPT = """A normalization document has sections for different kinds of items. Pick the section that describes the recorded item (judge by what the item is, its head noun; tag numbers do not matter), or "NONE" if no section describes it."""

SPEC_CHOICE_PROMPT = """A data sheet row has an attribute label. A normalization document defines rules for the attributes listed below. Pick the one whose rules govern this label for this item — the same property under another name, singular/plural, abbreviation, a more specific name of it, or the kind of quantity the document's section for this item says the label is recorded as — or "NONE" if the document has no rules for this property. Do not pick an attribute just because one word of its name matches. An attribute whose name equals the label is listed first: pick it unless its rules are written only for a different kind of item than this row's item — then pick the attribute whose rules are for this item, or "NONE" if there is none. Judge the item by what it is (its head noun, its kind), not by tag numbers."""

READING_PROMPT = """You read one recorded value the way a normalization document says input values are read, before anything is written. Say what quantity the record states: one number and the unit it is in.
- Give the number in the unit the record (or its attribute label) states it in; do not convert it to another unit, conversion is done later. Only where the document itself defines how such a record is read (a code that stands for a size, a formula, a notation with its own meaning) give the number that reading yields, in the unit it yields.
- Where the record alone does not say something you need (no unit, a decimal comma, a code or notation), use only what the document says about reading such records, never your own assumption.
- Every unit or number you take from the document rather than from the record or its label is an assumption: give the exact sentence of the document that says it.
- If the written value must carry anything besides the number and its unit (a qualifier such as approx., nominal, min., a condition such as @ 40 °C, a prefix or suffix the record or the document's reading keeps), give it in "keep"; leave "keep" empty only when the number and unit are the whole value.
Return JSON: {"single": false if the record states a range, several quantities or no quantity, else true; "number": the number as a plain decimal, or null; "unit": the unit of that number as written in the record, the label or the document, or ""; "keep": what the written value must carry besides the number and its unit, or ""; "assumptions": [{"what": what you took from the document, "quote": the exact sentence of the document, copied verbatim}]}"""

TEXT_PROMPT = """You write one recorded value the way a normalization document requires. Apply ONLY the attribute's instructions and rules and the document's general rules; change nothing they do not ask for, add nothing the record does not contain. General rules (letter case, empty or placeholder values, how to write a value that cannot be mapped) apply unless the attribute's rules say otherwise.
Return JSON: {"value": the written value, "flag": true only when the rules say to keep the value as received and pass it for review, or to report it}, "applied": [every rule sentence you applied to this value, copied verbatim from the rules or general rules; [] if none applied]}."""

BLIND_TEXT_PROMPT = """You are an independent checker. From the document's rules alone, work out how this recorded value must be written: apply every rule that applies to it, and only those; keep what no rule changes; add nothing the record does not contain.
Return JSON: {"value": the value as the rules require it to be written, "flag": true only when the rules say to keep the value as received and pass it for review, or to report it}."""


class SpecRunner:
    def __init__(self, doc: str, model: ChatOpenAI, user_rules: bool = True):
        folder = OUTPUT_DIR / doc
        self.doc = doc
        self.specs = json.loads((folder / "attribute_spec.json").read_text(encoding="utf-8"))["specs"]
        # паспорта, перекомпилированные с правилами пользователей (user_rules.json): заменяют
        # паспорт документа с тем же атрибутом, паспорт нового атрибута добавляется
        # user_rules=False — паспорта документа как есть: так проверяется только что собранный
        # паспорт, не запуская при этом пересборку остальных
        if user_rules:
            from .modules.rule_store import apply_overrides
            self.specs = apply_overrides(doc, self.specs, model)
        groups = json.loads((folder / "rule_groups.json").read_text(encoding="utf-8"))
        self.global_rules = "\n".join(r for g in groups if g.get("type") == "global"
                                      for r in g.get("rules", []))[:GLOBAL_CHARS]
        # разделы по видам изделий: какие у изделия атрибуты и как каждый из них пишется
        self.classes = {g["attribute"]: "\n".join(g.get("rules", [])) for g in groups if g.get("kind") == "class"}
        # корни обозначений (keyword → root) — данные извлечения, а не текст правил: в тексте
        # документа корень может стоять в двух написаниях, а в прозе его может не быть вовсе
        load = lambda name: json.loads((folder / name).read_text(encoding="utf-8")) if (folder / name).exists() else {}
        self.roots = load("deterministic_rules.json")
        self.root_order = load("deterministic_order.json")
        self.root_owner = load("deterministic_attributes.json")
        # перечни соответствий документа (table_roles.json: синонимы и реестры кодов, каждая пара
        # сверена с текстом при экстракции). Длинный перечень модель переписывает в спецификацию
        # неполно — у EMD из трёх таблиц материалов в список попала одна, — поэтому точная форма
        # ищется и здесь, по владельцу перечня
        self.table_lists = [item for item in (load("table_roles.json") or {}).get("lists", [])
                            if item.get("role") in ("synonyms", "register") and item.get("pairs")]
        self.model = model
        self._spec_cache: dict[str, dict | None] = {}
        self._lock = threading.Lock()

    def clone_with(self, spec: dict) -> "SpecRunner":
        """Тот же исполнитель с одним заменённым (или добавленным) паспортом и пустым кэшем —
        для предпросмотра правила: как бы легли строки, не трогая рабочий исполнитель."""
        clone = object.__new__(SpecRunner)
        clone.__dict__.update(self.__dict__)
        clone.specs = [s for s in self.specs if s["attribute"] != spec["attribute"]] + [spec]
        clone._spec_cache = {}
        clone._lock = threading.Lock()
        return clone

    def spec_named(self, name: str) -> dict | None:
        """Паспорт по точному имени атрибута (без учёта регистра и пробелов)."""
        key = _norm_attr(name)
        return next((s for s in self.specs if _norm_attr(s["attribute"]) == key), None)

    # ------------------------------------------------------------------ поиск спецификации
    def spec_for(self, attribute: str, context: str = "", value: str = "") -> dict | None:
        key = _norm_attr(attribute)
        exact = [s for s in self.specs if key == _norm_attr(s["attribute"])]
        # одно имя ещё не значит одно поле: «Length» у перчаток — длина манжеты, у болта — длина
        # болта, а по имени бралась спецификация болта (12 in → 304.8 mm вместо 305 mm); и
        # спецификация с этим именем может быть написана для другого изделия, чем строка. Поэтому
        # при изделии в строке выбирает модель, видя изделие; спецификация с тем же именем идёт
        # первой. Без изделия решать нечем — точное имя решает само
        if exact and not context.strip():
            return exact[0]
        # выбирать среди тех, кто на подпись претендует (именем или алиасом), если их несколько:
        # из всех спецификаций сразу модель не узнала в «HHB» болт и взяла покрытие перчаток
        claimants = [s for s in self.specs
                     if key in {_norm_attr(n) for n in [s["attribute"], *(s.get("aliases") or [])]}]
        pool = claimants if len(claimants) > 1 else self.specs
        candidates = exact + [s for s in pool if s not in exact]
        row_class = self._class_for(context)
        # значение тоже решает: «Rating: 600#» — класс давления, «Rating: PN40» — номинальное давление
        cache_key = f"spec|{key}|{row_class or _norm_attr(context)}|{value}"
        with self._lock:
            if cache_key in self._spec_cache:
                return self._spec_cache[cache_key]
        # имя в документе и подпись в таблице — разные тексты («End connections» / «End connection»,
        # «Head» насоса — это «Long length» по разделу класса): какой спецификации подпись
        # принадлежит, решает модель — по именам, алиасам, изделию строки и разделу его класса
        names = list(dict.fromkeys(s["attribute"] for s in candidates))
        schema = {"title": "spec", "type": "object", "additionalProperties": False,
                  "properties": {"attribute": {"type": "string", "enum": names + ["NONE"]}},
                  "required": ["attribute"]}
        listing = "\n".join(f"- {s['attribute']}" + (f" (also: {', '.join(s['aliases'][:6])})" if s.get("aliases") else "")
                            for s in candidates)
        class_text = (f"\n\nThe document's section for this kind of item ({row_class}):\n{self.classes[row_class][:3000]}"
                      if row_class else "")
        answer = self.model.with_structured_output(schema, method="json_schema", strict=True).invoke([
            {"role": "system", "content": SPEC_CHOICE_PROMPT},
            {"role": "user", "content": f"{_ctx(context)}Label: {attribute}\nRecorded value: {value}{class_text}\n\nAttributes of the document:\n{listing}"}])
        spec = next((s for s in self.specs if s["attribute"] == answer["attribute"]), None)
        with self._lock:
            self._spec_cache[cache_key] = spec
        return spec

    def _class_for(self, context: str) -> str | None:
        """Раздел документа для изделия строки (группы kind="class"), выбранный моделью; кэш по контексту."""
        if not self.classes or not context:
            return None
        cache_key = f"class|{context.strip().lower()}"
        with self._lock:
            if cache_key in self._spec_cache:
                return self._spec_cache[cache_key]
        names = list(self.classes)
        schema = {"title": "cls", "type": "object", "additionalProperties": False,
                  "properties": {"section": {"type": "string", "enum": names + ["NONE"]}}, "required": ["section"]}
        answer = self.model.with_structured_output(schema, method="json_schema", strict=True).invoke([
            {"role": "system", "content": CLASS_PROMPT},
            {"role": "user", "content": f"Item: {context}\n\nSections:\n" + "\n".join(
                f"- {n}: {t[:200]}" for n, t in self.classes.items())}])
        chosen = answer["section"] if answer["section"] in self.classes else None
        with self._lock:
            self._spec_cache[cache_key] = chosen
        return chosen

    def _variant(self, spec: dict, context: str, value: str = "") -> dict:
        """Вариант спецификации для изделия строки («Size» клапана — NPS, кабеля — жилы × сечение).

        Какому варианту принадлежит изделие, решает модель (выбор из «applies_to» вариантов или
        «none of these» — тогда общая часть); ответ кэшируется на пару (атрибут, контекст).
        """
        variants = spec.get("variants") or []
        if not variants or not (context or value):
            return spec
        # варианты бывают по изделию и по самому значению («below 1000 V» / «from 1000 V»)
        key = f"variant|{spec['attribute']}|{context.strip().lower()}|{value}"
        with self._lock:
            cached = self._spec_cache.get(key, "?")
        if cached == "?":
            labels = [v["applies_to"] for v in variants]
            schema = {"title": "variant", "type": "object", "additionalProperties": False,
                      "properties": {"variant": {"type": "string", "enum": labels + ["none of these"]}},
                      "required": ["variant"]}
            answer = self.model.with_structured_output(schema, method="json_schema", strict=True).invoke([
                {"role": "system", "content": VARIANT_PROMPT},
                {"role": "user", "content": f"Item: {context or '(not given)'}\nAttribute: {spec['attribute']}\nRecorded value: {value}\n\nVariants:\n"
                                            + "\n".join(f"- {l}" for l in labels)}])
            cached = next((v for v in variants if v["applies_to"] == answer["variant"]), None)
            with self._lock:
                self._spec_cache[key] = cached
        if not cached:
            return spec
        merged = {**spec, **{k: v for k, v in cached.items()
                             if k != "applies_to" and v not in (None, "", [])}}
        merged["variants"] = []
        merged["applies_to"] = cached["applies_to"]
        return merged

    def _roots_for(self, attribute: str, spec) -> str:
        """Корни обозначений, чей атрибут-владелец — этот атрибут (по подписи или по спецификации)."""
        names = [attribute] + ([spec["attribute"]] if spec else [])
        lines = []
        for keyword, root in self.roots.items():
            owner = self.root_owner.get(keyword, "")
            if any(_attribute_matches(n, owner) or _attribute_matches(n, keyword) for n in names if n):
                lines.append(f"{keyword} -> {root}" + (" (number first)" if self.root_order.get(keyword) else ""))
        return "\n".join(lines)

    # ------------------------------------------------------------------ исполнение
    def run(self, attribute: str, value: str, context: str = "") -> tuple[str, bool, str]:
        value = str(value).strip()
        spec = self.spec_for(attribute, context, value)
        if spec is None:
            return self._text(None, attribute, value, context)
        spec = self._variant(spec, context, value)
        written, flag, note = self._run(spec, attribute, value, context)
        # регистр букв записанного значения: значение списка пишется дословно, и «писать
        # заглавными» иначе не исполнилось бы ни в одном пути. Поле ставит и документ («Capitals,
        # no spaces»), не только правило пользователя — подпись по тому, есть ли его строки
        case = spec.get("letter_case")
        if written and case in ("upper", "lower"):
            from .modules.rule_store import USER_RULE_MARK
            written = written.upper() if case == "upper" else written.lower()
            by_user = USER_RULE_MARK in str(spec.get("rules", ""))
            note += (f"; регистр: {'заглавные' if case == 'upper' else 'строчные'}"
                     + (" (паспорт с правилами пользователя)" if by_user else " (по паспорту документа)"))
        # какой паспорт атрибута сработал на строке — первым после уровня уверенности
        level, _, rest = note.partition("; ")
        where = f"паспорт «{spec['attribute']}»" + (f", вариант «{spec['applies_to']}»" if spec.get("applies_to") else "")
        return written, flag, f"{level}; {where}" + (f"; {rest}" if rest else "")

    def _run(self, spec: dict, attribute: str, value: str, context: str) -> tuple[str, bool, str]:
        if not value:
            return self._text(spec, attribute, value, context)
        if spec["kind"] == "list" and spec["values"]:
            return self._list(spec, attribute, value, context)
        if spec["kind"] == "quantity":
            done = self._quantity(spec, attribute, value)
            if done:
                return done
        return self._text(spec, attribute, value, context)

    def _list(self, spec, attribute, value, context):
        """Значение из списка спецификации, с уровнем уверенности по доказательствам.

        Высокий — то, что код проверил как факт: форма есть в таблице документа, модель сослалась
        на правило и цитата есть в тексте, модель сказала «та же формулировка» и состав слов
        совпал. Средний — суждение модели (эквивалентная запись того же значения), которое
        принимается, только если его вслепую повторил независимый второй вызов. Низкий («не
        уверена») и разошедшиеся — жёлтым, с предложенным значением. Код смысл не судит.
        """
        key = form_key(value)
        for entry in spec["values"]:
            if key in {form_key(entry["value"])} | {form_key(a) for a in entry["accepted"]}:
                return entry["value"], False, f"высокая: форма из таблицы документа; «{value}» → «{entry['value']}»"
        found = self._document_form(spec, attribute, key)
        if found:
            return found, False, f"высокая: форма из перечня документа; «{value}» → «{found}»"
        choices = list(dict.fromkeys(e["value"] for e in spec["values"])) + ["UNMATCHED"]
        schema = {"title": "choice", "type": "object", "additionalProperties": False,
                  "properties": {"choice": {"type": "string", "enum": choices},
                                 "basis": {"type": "string", "enum": ["same wording", "rule", "equivalent", "unsure"]},
                                 "rule_quote": {"type": "string"}},
                  "required": ["choice", "basis", "rule_quote"]}
        listing = "\n".join(f"- {e['value']} (also written: {'; '.join(e['accepted'])})" for e in spec["values"])
        question = f"{_ctx(context)}Attribute: {attribute}\nRecorded value: {value}\n\nList:\n{listing}\n\nRules:\n{spec['rules']}"
        answer = self.model.with_structured_output(schema, method="json_schema", strict=True).invoke([
            {"role": "system", "content": CHOICE_PROMPT}, {"role": "user", "content": question}])
        choice, basis = answer["choice"], answer.get("basis")
        entry = next((e for e in spec["values"] if e["value"] == choice), None)
        if entry is None:
            # значения в списке нет: как его записать (оставить, пометить, «UNMAPPED: …» или по
            # формату правила, если список неполный), говорят правила — но не кодом из списка
            result = self._text(spec, attribute, value, context,
                                note="The recorded value is NOT one of the list values: do not write any list value or a code in their style.")
            if result[0] in choices[:-1]:
                return value, True, "низкая: значения нет в списке, код не присваивается"
            return result[0], result[1] or bool(spec.get("closed")), "нет в списке; " + result[2]
        # основание, которое код может проверить как факт
        if basis == "same wording" and any(_wording_matches(value, f) for f in [entry["value"]] + entry["accepted"]):
            return choice, False, f"высокая: та же формулировка (сверено); «{value}» → «{choice}»"
        quote = _norm_text(answer.get("rule_quote") or "")
        if basis == "rule" and len(quote) >= 12 and quote in _norm_text(spec["rules"]):
            return choice, False, (f"высокая: правило (цитата сверена); «{value}» → «{choice}»; "
                                   f"правило: {_cite(answer.get('rule_quote') or '', spec)}")
        if basis == "unsure":
            return value, True, f"низкая: предложено «{choice}»"
        # суждение модели без проверяемого факта — повторить вслепую
        blind = self._blind_choice(choices, listing, question)
        if blind == choice:
            return choice, False, f"средняя: суждение модели, сверено вслепую; «{value}» → «{choice}» как {basis or 'equivalent'}"
        return value, True, f"разошлись: «{choice}» / «{blind}» — на проверку"

    def _document_form(self, spec, attribute: str, key: str) -> str | None:
        """Запись — форма из перечня документа для этого атрибута: одна запись на все совпадения.

        Владелец перечня назван при экстракции (имя атрибута, чьи значения перечислены); перечень
        берётся, если владелец совпадает со спецификацией строки или с её подписью. Форма, ведущая
        к двум разным записям, ничего не решает.
        """
        names = [spec["attribute"], attribute] + list(spec.get("aliases") or [])
        hits = set()
        for item in self.table_lists:
            owner = str(item.get("attribute") or item.get("group") or "")
            if not any(_attribute_matches(n, owner) for n in names if n):
                continue
            for pair in item["pairs"]:
                if key in (form_key(pair.get("form", "")), form_key(pair.get("value", ""))):
                    hits.add(pair["value"])
        return hits.pop() if len(hits) == 1 else None

    def _blind_choice(self, choices: list[str], listing: str, question: str) -> str:
        """Независимый выбор: другой вопрос, варианты в обратном порядке, первого ответа не видит."""
        schema = {"title": "check", "type": "object", "additionalProperties": False,
                  "properties": {"value": {"type": "string", "enum": list(reversed(choices))}}, "required": ["value"]}
        reordered = question.replace(listing, "\n".join(reversed(listing.split("\n"))))
        answer = self.model.with_structured_output(schema, method="json_schema", strict=True).invoke([
            {"role": "system", "content": BLIND_CHOICE_PROMPT}, {"role": "user", "content": reordered}])
        return answer["value"]

    def _quantity(self, spec, attribute, value):
        reading = self._read(spec, attribute, value)
        if reading is None or not reading["single"] or reading["number"] is None:
            return None  # диапазон, перечень или не прочитано — по инструкции
        unit_text = reading["unit"]
        # код пишет только число и единицу; если в значении должно остаться что-то ещё
        # («approx. 300 kW», «ISO VG 220 base» → «220 cSt @ 40 °C»), это видит модель при чтении,
        # и такое значение пишет она по правилам
        if reading["keep"]:
            return None
        target = spec.get("unit") or ""
        if not target and unit_text:
            return None  # единицу записи документ не задал, а у записи она есть — по инструкции
        if target and not _pint(target, self.model):
            return None  # в поле единицы не единица («VAC/VDC or kV») — спецификацию код не исполняет
        number = reading["number"]
        converted = False
        # что сделал код — для пояснения строки
        ops = [f"прочитано {_num(number)}{(' ' + unit_text) if unit_text else ''}"]
        source = _pint(unit_text, self.model) if unit_text else ""
        if unit_text and not source:
            return None  # единица записи не распознана — по инструкции
        if source and target:
            # коэффициент документа подходит, только если его единица — та же, что у записи;
            # нераспознанная единица («GPM (US)») не совпадает ни с чем, а не со всем подряд
            # запись не по схеме (нет «from»/«factor») код не исполняет, а пропускает
            conversions = [c for c in spec.get("conversions") or []
                           if isinstance(c, dict) and c.get("from") and isinstance(c.get("factor"), (int, float))]
            factor = next((c for c in conversions
                           if form_key(c["from"]) == form_key(unit_text or "")
                           or (_pint(c["from"], self.model) and same_unit(_pint(c["from"], self.model), source))), None)
            target_pint = _pint(target, self.model)
            if factor:
                number = number * float(factor["factor"]) + float(factor.get("offset") or 0)
                converted = float(factor["factor"]) != 1 or bool(factor.get("offset"))
                if converted:
                    ops.append(f"пересчёт {unit_text} → {target} по коэффициенту документа "
                               f"(×{_num(float(factor['factor']))}"
                               + (f" {float(factor['offset']):+g}" if factor.get("offset") else "") + ")")
            elif target_pint and unit_mismatch(source, target_pint):
                result = convert_value(number, source, target_pint)
                if result is None:
                    return None
                number, converted = result, True
                ops.append(f"пересчёт {unit_text} → {target} (pint, коэффициента в документе нет)")
        # голое число (ни в записи, ни в подписи единицы нет) не пересчитывается; единица документа
        # ставится только как предложение на проверку — её неоткуда взять; число пишется по формату
        bare = not source
        series = [float(s) for s in spec.get("series") or []]
        if series:
            number = min(series, key=lambda s: abs(s - number))
            ops.append(f"ближайшее из ряда документа: {_num(number)}")
        pattern, locale = _number_pattern(spec)
        if spec.get("number_format") == "fraction":
            text = _fraction(number)  # дюймовые размеры: «1-1/8», «3/4»
            ops.append("дробью (по документу)")
        elif not pattern and number != int(number):
            return None  # запись числа документ не задал — по инструкции документа
        elif not valid_number_pattern(pattern) or not valid_locale(locale):
            return None  # шаблон не по синтаксису ICU — спецификацию код не исполняет
        else:
            # шаблон ICU исполняет Babel: «0.0» — ровно один знак, «0.#» — до одного без нулей в конце
            if spec.get("rounding_scope") == "converted" and not converted:
                pattern = keep_digits(pattern or "0", number)  # документ округляет только пересчитанное
                ops.append("уже в единице документа — знаки как в записи")
            else:
                ops.append(f"запись числа «{pattern or '0'}»" + (f", локаль {locale}" if locale != "en" else ""))
            text = format_number(number, pattern or "0", locale, spec.get("rounding") or "half_up")
        unit = target or unit_text
        written = f"{text}{' ' if spec.get('unit_space', True) and unit else ''}{unit}"
        done = "; ".join(ops)
        if bare and target:
            # единицы нет ни в записи, ни в подписи, и документ не говорит, какая она: что число
            # именно в единице записи, код не знает — предложение уходит на проверку
            return written, True, f"низкая: в записи нет единицы, предложено в единице документа ({target}); {done}"
        if reading["unverified"]:
            return written, True, ("низкая: допущение чтения без цитаты в документе — "
                                   + "; ".join(reading["unverified"])[:200] + f"; {done}")
        basis = "; прочитано по документу: " + " | ".join(reading["assumptions"])[:200] if reading["assumptions"] else ""
        return written, False, f"высокая: величина, пересчёт и формат кодом; {done}" + basis

    def _read(self, spec, attribute, value) -> dict | None:
        """Что записано: число и его единица — прочитанные моделью так, как велит документ.

        Чтение входа — не поле спецификации: правил чтения бесконечно много (число без единицы —
        это мм, десятичная запятая, код ремня B75 — это дюймы плюс поправка сечения). Модель
        читает запись вместе с правилами и на каждое допущение, взятое из документа, а не из
        записи, даёт дословную цитату; код ищет цитату в тексте. Пересчёт, округление и запись —
        код. Кэш на (спецификация, подпись, значение)."""
        rules = spec.get("rules") or ""
        cache_key = f"read|{spec['attribute']}|{spec.get('applies_to') or ''}|{attribute}|{value}"
        with self._lock:
            if cache_key in self._spec_cache:
                return self._spec_cache[cache_key]
        reading = None
        try:
            answer = self.model.invoke([
                {"role": "system", "content": READING_PROMPT},
                # общие положения первыми: одинаковое начало запроса кэшируется у провайдера
                {"role": "user", "content": f"General rules of the document:\n{self.global_rules}\n\n"
                                            f"Rules of the document for this attribute ({spec['attribute']}):\n{rules}\n\n"
                                            f"Attribute label: {attribute}\nRecorded value: {value!r}"}],
                response_format={"type": "json_object"})
            data = json.loads(str(answer.content))
            source = _norm_text(rules + "\n" + self.global_rules)
            verified, unverified = [], []
            for item in data.get("assumptions") or []:
                quote = _norm_text(item.get("quote") or "")
                what = str(item.get("what") or "").strip()
                (verified if len(quote) >= 8 and quote in source else unverified).append(
                    f"{what} («{str(item.get('quote') or '')[:120]}»)")
            number = data.get("number")
            reading = {"single": bool(data.get("single", True)),
                       "number": float(number) if isinstance(number, (int, float)) else None,
                       "unit": str(data.get("unit") or "").strip(),
                       "keep": str(data.get("keep") or "").strip(),
                       "assumptions": verified, "unverified": unverified}
        except Exception:
            reading = None
        with self._lock:
            self._spec_cache[cache_key] = reading
        return reading

    def _text(self, spec, attribute, value, context="", note=""):
        if spec is None:
            body = "The document has no rules for this attribute; only its general rules apply."
        else:
            body = (f"Instructions: {spec.get('instructions', '')}\nWhen to keep and flag: {spec.get('flag_when') or 'never'}\n\n"
                    f"Rules of the document for this attribute:\n{spec['rules']}")
        roots = self._roots_for(attribute, spec)
        if roots:
            body += ("\n\nDesignation roots defined by the document (keyword -> root; the number of the "
                     "record is written right after the root, or before it where marked; where the rules "
                     "spell a root differently, this table wins):\n" + roots)
        question = (f"{_ctx(context)}Attribute: {attribute}\nRecorded value: {value!r}\n{note}\n\n{body}\n\n"
                    f"General rules of the document:\n{self.global_rules}")
        first = self._ask_text_full(TEXT_PROMPT, question)
        if first is None:
            return value, True, "низкая: ответ модели не разобран"
        written, flag, applied = first
        # какие правила модель применила — только цитаты, которые код нашёл в тексте правил
        source = (spec.get("rules", "") if spec else "") + "\n" + self.global_rules
        cited = [_cite(q, spec) for q in applied if len(_norm_text(q)) >= 8 and _norm_text(q) in _norm_text(source)][:3]
        rules_note = ("; правила: " + "; ".join(cited)) if cited else "; применённое правило не процитировано"
        if flag:
            return written, True, "документ велит оставить и передать на проверку" + rules_note
        # ответ по инструкции — суждение модели, проверить его фактом нечем: повторить вслепую
        second = self._ask_text(BLIND_TEXT_PROMPT, question)
        if second is not None and _same_text(second[0], written):
            return written, False, "средняя: по инструкции, сверено вслепую" + rules_note
        other = second[0] if second else "—"
        return written, True, f"разошлись: «{written}» / «{other}» — на проверку" + rules_note

    def _ask_text(self, prompt: str, question: str) -> tuple[str, bool] | None:
        full = self._ask_text_full(prompt, question)
        return full[:2] if full else None

    def _ask_text_full(self, prompt: str, question: str) -> tuple[str, bool, list[str]] | None:
        """(значение, пометка, применённые правила дословно — как их назвала модель)."""
        answer = self.model.invoke([{"role": "system", "content": prompt},
                                    {"role": "user", "content": question}],
                                   response_format={"type": "json_object"})
        try:
            data = json.loads(str(answer.content))
            applied = [str(q) for q in data.get("applied") or [] if isinstance(q, str)]
            return str(data.get("value", "")).strip(), bool(data.get("flag")), applied
        except Exception:
            return None


def same_unit(left: str, right: str) -> bool:
    """Одна ли это единица для pint; нераспознанная — не та же."""
    try:
        return _unit(left) == _unit(right)
    except Exception:
        return False


def _same_text(left: str, right: str) -> bool:
    """Два вывода совпали: различие только в лишних пробелах не в счёт."""
    return " ".join(str(left).split()) == " ".join(str(right).split())


def _norm_text(text: str) -> str:
    return " ".join(str(text).lower().replace("’", "'").replace('"', "").replace("'", "").split())


def _num(number: float) -> str:
    """Число для пояснения: без хвоста «.0» и без двадцати знаков после точки."""
    return f"{number:.6g}"


def _cite(quote: str, spec: dict | None) -> str:
    """Цитата правила для пояснения, с источником, если это правило пользователя."""
    from .modules.rule_store import USER_RULE_MARK
    text = " ".join(str(quote).replace(USER_RULE_MARK, "").split())
    short = text if len(text) <= 160 else text[:157] + "…"
    user_lines = [line for line in str((spec or {}).get("rules", "")).splitlines() if line.startswith(USER_RULE_MARK)]
    mine = any(_norm_text(text) in _norm_text(line) for line in user_lines)
    return f"«{short}»" + (" (правило пользователя)" if mine else "")


def _ctx(context: str) -> str:
    return f"Row context: {context}\n" if context else ""


_PINT_CACHE: dict[str, str] = {}


def _pint(unit_text: str, model) -> str:
    """Имя единицы для pint по её записи в документе («m³», «barg», «°C») — через разбор модели."""
    if unit_text not in _PINT_CACHE:
        parsed = parse(f"1 {unit_text}", "", model)
        _PINT_CACHE[unit_text] = parsed.quantities[0].unit if parsed and parsed.quantities else ""
    return _PINT_CACHE[unit_text]


def _fraction(number: float) -> str:
    """Число дробью со знаменателем-степенью двойки до 64: 1.125 → «1-1/8», 0.75 → «3/4»."""
    from fractions import Fraction
    frac = Fraction(number).limit_denominator(64)
    whole, rest = divmod(frac.numerator, frac.denominator)
    if rest == 0:
        return str(whole)
    part = f"{rest}/{frac.denominator}"
    return f"{whole}-{part}" if whole else part


def _number_pattern(spec: dict) -> tuple[str, str]:
    """(шаблон ICU, локаль) спецификации. Спецификации, собранные до шаблонов, хранят
    decimals / max_decimals / decimal_separator — они переводятся в тот же шаблон."""
    if "number_pattern" in spec:
        return spec.get("number_pattern") or "", spec.get("locale") or "en"
    locale = "ru" if spec.get("decimal_separator") == "," else "en"
    if spec.get("decimals") is not None:
        digits = int(spec["decimals"])
        return ("0." + "0" * digits if digits else "0"), locale
    if spec.get("max_decimals") is not None:
        digits = int(spec["max_decimals"])
        return ("0." + "#" * digits if digits else "0"), locale
    return "", locale


# ---------------------------------------------------------------------- встраивание в API
SPEC_WORKERS = 6
_RUNNERS: dict[str, SpecRunner] = {}
_RUNNERS_LOCK = threading.Lock()


def has_spec(rule_set) -> bool:
    """Есть ли у документа скомпилированная спецификация (attribute_spec.json)."""
    return (OUTPUT_DIR / rule_set.name / "attribute_spec.json").exists()


def get_runner(doc: str, model) -> SpecRunner:
    """Исполнитель документа (один на документ: в нём кэш сопоставлений подписи с паспортом)."""
    with _RUNNERS_LOCK:
        runner = _RUNNERS.get(doc)
        if runner is None:
            runner = _RUNNERS[doc] = SpecRunner(doc, model)
    return runner


def invalidate_runner(doc: str) -> None:
    """Забыть исполнитель документа: паспорта поменялись (добавлено или снято правило)."""
    with _RUNNERS_LOCK:
        _RUNNERS.pop(doc, None)


def run_spec_rule_set(data_frame, model, rule_set, progress=None):
    """Колонка одного документа по спецификации — тот же контракт, что у graph.run_single_rule_set.

    Возвращает таблицу с колонками ROW_COLUMNS: значение, пометка на проверку («Ambiguous»), что
    сработало («Applied» — уровень уверенности и его основание). Строка изделия, которое документ
    не описывает, остаётся пустой и не помечается — как в прежнем конвейере. Исполнитель на
    документ создаётся один раз: в нём кэш сопоставлений подписи со спецификацией.
    """
    import pandas as pd

    from .pipeline.frames import _row_context
    from .pipeline.graph import ROW_COLUMNS
    from .pipeline.parsing import set_parser_model
    from .pipeline.subjects import _document_covers_row

    set_parser_model(model)
    runner = get_runner(rule_set.name, model)

    def one(i: int) -> dict:
        attribute = str(data_frame.loc[i, "Attribute"])
        value = str(data_frame.loc[i, "Value"])
        context = _row_context(data_frame, i)
        row = {"Attribute": attribute, "Value": value, "Context": context, "Deterministic": None}
        if context and not _document_covers_row(rule_set, context, model):
            return {**row, "Rules_Value": "", "Ambiguous": False,
                    "Applied": "документ не распространяется на это изделие"}
        try:
            written, flag, note = runner.run(attribute, value, context)
        except Exception as e:  # строка не должна ронять всю колонку
            logger.warning("  [SPEC] %r / %r: %s", attribute, value, e)
            written, flag, note = value, True, "ответ не получен — оставлено как записано"
        finally:
            if progress is not None:
                progress.step()
        return {**row, "Rules_Value": written, "Ambiguous": bool(flag), "Applied": note}

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=SPEC_WORKERS) as pool:
        rows = list(pool.map(one, range(len(data_frame))))
    return pd.DataFrame(rows, columns=ROW_COLUMNS)
