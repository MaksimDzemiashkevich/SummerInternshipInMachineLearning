"""Разбор значения отдельным вызовом модели: из чего запись (или ответ) состоит — вид значения,
числа с единицами, корень и номер обозначения.

Строку читает модель, а не регулярные выражения: написаний одного и того же («0.5-inch»,
«2 1/2"», «600#», «10...20 mm») слишком много. Вызов отдельный от нормализации: проверка ловит
ошибки нормализующей модели, и разбор, выданный тем же ответом, мог бы ошибиться с ним
согласованно. Нет модели или ответ не разобран — None, и проверки ничего не отклоняют.

Parser — разборщик с моделью и своим кэшем; его создаёт тот, кто разбирает (исполнитель
паспортов). Функции parse / parse_many / set_parser_model — разборщик по умолчанию, которым
пользуются проверки запасного пути (guards, postprocess, answers): модель ему задаёт
run_single_rule_set.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ..modules.system_prompts import PARSE_PROMPT
from ..modules.tools import unit_mismatch
from .common import Memo, logger


@dataclass
class Quantity:
    number: str          # как записано: «0.500», «7,62»
    value: float | None  # то же число десятичной дробью
    unit: str            # стандартное имя для pint: «inch», «psig»; «» — единицы нет
    unit_text: str       # как записано: «-inch», «in.», «"», «bar(g)»
    space: bool          # пробел между числом и единицей
    reference: str = ""  # «gauge» / «absolute» для давления с такой пометкой, иначе «»


@dataclass
class Designation:
    root: str            # как записано: «CL», «TRIM», «#»
    number: str          # как записано: «600», «12A»
    space: bool
    number_first: bool


@dataclass
class Parsed:
    kind: str
    quantities: list[Quantity] = field(default_factory=list)
    designation: Designation | None = None
    label_unit: str = ""

    @property
    def single(self) -> Quantity | None:
        """Единственная величина записи — только если запись и есть одна величина."""
        return self.quantities[0] if self.kind == "quantity" and len(self.quantities) == 1 else None


def _as_parsed(data: dict) -> Parsed:
    """Ответ модели (JSON) → Parsed; неизвестные поля и нечисловые значения отбрасываются."""
    quantities = []
    for q in data.get("quantities") or []:
        if not isinstance(q, dict):
            continue
        try:
            value = float(q.get("value")) if q.get("value") is not None else None
        except (TypeError, ValueError):
            value = None
        reference = str(q.get("reference") or "").lower()
        quantities.append(Quantity(str(q.get("number", "")), value, str(q.get("unit") or ""),
                                   str(q.get("unit_text") or ""), bool(q.get("space")),
                                   reference if reference in ("gauge", "absolute") else ""))
    d = data.get("designation")
    designation = (Designation(str(d.get("root", "")), str(d.get("number", "")),
                               bool(d.get("space")), bool(d.get("number_first")))
                   if isinstance(d, dict) and d.get("number") else None)
    return Parsed(str(data.get("kind") or "text"), quantities, designation,
                  str(data.get("label_unit") or ""))


class Parser:
    """Разбор значений моделью; одна и та же строка с той же подписью разбирается один раз."""

    def __init__(self, model=None, memo: Memo | None = None):
        self.model = getattr(model, "bound", model)  # модель без привязанных инструментов
        self._memo = memo or Memo("разбор значений")

    def parse(self, text: str, label: str = "", model=None) -> Parsed | None:
        """Разбор строки (с подписью атрибута — для записи) или None, если разобрать нечем."""
        text = str(text).strip()
        model = getattr(model, "bound", model) if model is not None else self.model
        if model is None:
            return None
        return self._memo.remember((text, str(label).strip()), lambda: self._ask(model, text, label))

    def parse_many(self, items: list[tuple[str, str]], model=None, workers: int = 8) -> None:
        """Заранее разобрать много строк параллельно: дальше они берутся из кэша."""
        todo = list(dict.fromkeys(i for i in items if (str(i[0]).strip(), str(i[1]).strip()) not in self._memo))
        if todo:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(lambda item: self.parse(item[0], item[1], model), todo))

    @staticmethod
    def _ask(model, text: str, label: str) -> Parsed | None:
        """Один вызов модели с PARSE_PROMPT; ошибка сети или неразборчивый ответ — None."""
        if not text:
            return Parsed("empty")
        user = (f"Attribute label: {label}\n" if label else "") + f"Value: {text}"
        try:
            raw = str(model.invoke([{"role": "system", "content": PARSE_PROMPT},
                                    {"role": "user", "content": user}]).content).strip()
            return _as_parsed(json.loads(raw.strip("`").removeprefix("json").strip()))
        except Exception as e:
            logger.warning("  [PARSE] %r не разобран: %s", text, e)
            return None


# разборщик по умолчанию — для проверок запасного пути
_DEFAULT = Parser()


def set_parser_model(model) -> None:
    """Модель разборщика по умолчанию (запасной путь задаёт её перед прогоном документа)."""
    _DEFAULT.model = getattr(model, "bound", model)


def parse(text: str, label: str = "", model=None) -> Parsed | None:
    """Разбор строки разборщиком по умолчанию (см. Parser.parse)."""
    return _DEFAULT.parse(text, label, model)


def parse_many(items: list[tuple[str, str]], model=None, workers: int = 8) -> None:
    """Заранее разобрать много строк разборщиком по умолчанию (см. Parser.parse_many)."""
    _DEFAULT.parse_many(items, model, workers)


def is_measure(unit: str) -> bool:
    """Единица измерения, а не пустое место: «count» у pint безразмерный, как и отсутствие единицы."""
    return bool(unit) and unit_mismatch(unit, "count")


def units_differ(left: Quantity, right: Quantity) -> bool:
    """Разные ли единицы у двух величин: pint по стандартным именам плюс отметка давления.

    Манометрическое и абсолютное давление различаются, только если отметка есть у обеих
    сторон: «barg» → «bara» — подмена, а «bar», «barg» и «bar(g)» — одна единица. Что в записи
    отметка, а что сама единица, решает разбор модели, а не список суффиксов в коде.
    """
    if left.reference and right.reference and left.reference != right.reference:
        return True
    return unit_mismatch(left.unit, right.unit)


def recorded_quantity(parsed: Parsed | None) -> Quantity | None:
    """Величина записи: её собственная, а у голого числа — число с единицей из подписи."""
    if parsed is None:
        return None
    if parsed.single and parsed.single.unit:
        return parsed.single
    if parsed.kind == "number" and parsed.label_unit and parsed.quantities:
        q = parsed.quantities[0]
        return Quantity(q.number, q.value, parsed.label_unit, "", False)
    return None


def recorded_unit(parsed: Parsed | None) -> str:
    """Единица записи: её собственная, а у голого числа — из подписи атрибута."""
    if parsed is None:
        return ""
    if parsed.single and parsed.single.unit:
        return parsed.single.unit
    if parsed.kind == "number":
        return parsed.label_unit
    return ""
