"""Инструменты модели: пересчёт единиц и правка разделителей.

Пересчёт единиц — один инструмент на все случаи, а не список готовых пар: пар в стандартах
десятки (cm→in, mm→in, psi→MPa, °F→°C, kg/mm²…), и стоит паре не найтись, как модель хватает
ближайшую похожую (реальный случай: для сантиметров вызвала inch→mm) или считает в уме.
Разбор и пересчёт делает pint — библиотека единиц со своей таблицей и проверкой размерности,
поэтому бессмысленный пересчёт (длина в массу) возвращает ошибку, а не число.
"""

import re

import pint
from langchain.tools import tool

_UREG = pint.UnitRegistry()

# Как единицы пишут в датащитах, но не понимает pint.
_UNIT_ALIASES = (
    (r'(?<=\d)\s*"', " inch"),
    (r"(?<=\d)\s*'", " foot"),
    (r"°\s*F\b|(?<=\d)\s*F\b|\bdeg\s*F\b", " degF"),
    (r"°\s*C\b|(?<=\d)\s*C\b|\bdeg\s*C\b", " degC"),
    (r"\bin\.(?=\s|$)", " inch"),
    (r"\blbs\b", " lb"),
    (r"(?<=[a-zA-Z])2\b", "**2"),      # mm2, in2 → mm**2
    (r"(?<=[a-zA-Z])3\b", "**3"),
)


# Дюймовые доли в трубопроводной записи: знаменатель всегда степень двойки. Это же условие
# защищает от подмены обозначений вида «0,6/1 kV», где косая черта — не дробь.
_FRACTION_DENOMINATORS = "(?:2|4|8|16|32|64)"
_VULGAR_FRACTIONS = {"½": " 1/2", "¼": " 1/4", "¾": " 3/4", "⅛": " 1/8", "⅜": " 3/8",
                     "⅝": " 5/8", "⅞": " 7/8", "⅓": " 1/3", "⅔": " 2/3"}


def _expand_fractions(text: str) -> str:
    """«2 1/2 in» → «2.5 in», «3/4"» → «0.75"», «2½» → «2.5».

    Без этого pint читает «2 1/2» как 2/(1/2) и молча возвращает 1 дюйм: ошибки нет,
    а число чужое. Смешанная запись размера в дюймах — норма в трубопроводной арматуре.
    """
    for sign, plain in _VULGAR_FRACTIONS.items():
        text = text.replace(sign, plain)

    def mixed(match):
        whole, num, den = int(match.group(1)), int(match.group(2)), int(match.group(3))
        return f"{whole + num / den:g}"

    def simple(match):
        return f"{int(match.group(1)) / int(match.group(2)):g}"

    text = re.sub(rf"(?<![\d./])(\d+)\s*[-\s]\s*(\d+)/({_FRACTION_DENOMINATORS})(?![\d/])",
                  mixed, text)
    return re.sub(rf"(?<![\d./])(\d+)/({_FRACTION_DENOMINATORS})(?![\d/])", simple, text)


def _prepare(text: str) -> str:
    """Запись из таблицы → выражение, понятное pint: дроби, запятая-разделитель, ° и ″, mm2."""
    prepared = _expand_fractions(str(text).strip())
    prepared = re.sub(r"(?<=\d),(?=\d)", ".", prepared)
    for pattern, replacement in _UNIT_ALIASES:
        prepared = re.sub(pattern, replacement, prepared)
    return re.sub(r"\s+", " ", prepared).strip()


def _format(magnitude: float, decimals: int) -> str:
    """Число в строку: с заданным количеством знаков либо без хвостовых нулей."""
    if decimals >= 0:
        return f"{magnitude:.{decimals}f}"
    text = f"{magnitude:.6f}".rstrip("0").rstrip(".")
    return text or "0"


def unit_mismatch(left: str, right: str) -> bool:
    """Разные ли это единицы по существу. Не единицы (или нечитаемое) — не разные.

    Сравнивать написанием нельзя: «in.», «inch» и «in» — одно и то же, а «HB minimum» и
    «Cr-10Ni-Mo» вообще не единицы. Разбор и сравнение отдаём pint, он же знает синонимы.
    """
    def as_unit(text: str):
        # одинокие F и C для pint — фарад и кулон, а в датащите это градусы
        prepared = _prepare(text).strip()
        return _UREG.Unit({"F": "degF", "C": "degC"}.get(prepared, prepared))

    try:
        return as_unit(left) != as_unit(right)
    except Exception:
        return False


def is_unit(word: str) -> bool:
    """Читается ли слово как единица измерения («in», «kPa», «rpm»), а не как содержимое записи."""
    try:
        return not _UREG.Unit(_prepare(word)).dimensionless
    except Exception:
        return False


@tool
def convert_units(quantity: str, to_unit: str, decimals: int = -1) -> str:
    """Convert a physical quantity into another unit, exactly.

    `quantity` is the recorded value with its unit, copied as it stands: "7,62 cm", "300 psi",
    "1200 F", '6"', "25 lbs", "150 mm". A decimal comma, a degree sign, an inch mark and a
    squared unit written as "mm2" are all understood.
    `to_unit` is the unit wanted: "in", "mm", "MPa", "degC", "kg", "mm**2" and so on.
    `decimals` rounds the result to that many decimal places; leave it out for the exact value
    with trailing zeros dropped.

    Returns "<number> <to_unit>", or a line starting with "cannot convert" when the value has
    no unit, the unit is unknown, or the two units measure different things — in that case do
    not invent a number, treat the conversion as not possible.
    """
    prepared = _prepare(quantity)
    target = _prepare(to_unit)
    try:
        parsed = _UREG.Quantity(prepared)
    except Exception as e:
        return f"cannot convert: {quantity!r} is not a quantity pint can read ({e})"
    if getattr(parsed, "dimensionless", False) and not re.search(r"[a-zA-Z]", prepared):
        return f"cannot convert: {quantity!r} carries no unit"
    try:
        converted = parsed.to(target)
    except pint.DimensionalityError as e:
        return f"cannot convert: {e}"
    except Exception as e:
        return f"cannot convert: unit {to_unit!r} is not known ({e})"
    return f"{_format(converted.magnitude, decimals)} {to_unit}"


@tool
def remove_characters(text: str, characters: str) -> str:
    """Remove EVERY occurrence of the given characters from the text and return the result.

    Use this whenever a rule says to drop a separator or punctuation mark, instead of retyping
    the value: the removal is then applied to every occurrence, and nothing else changes.
    `characters` is the set of characters to drop, e.g. "-" or "-,".
    """
    if not characters or not any(ch in text for ch in characters):
        return f"no change: none of {characters!r} occurs in the text"
    return "".join(ch for ch in text if ch not in set(characters))


@tool
def replace_substring(text: str, find: str, replace_with: str) -> str:
    """Replace EVERY occurrence of `find` with `replace_with` in the text and return the result.

    Use this whenever a rule says to swap one separator, word or symbol for another (for example
    a hyphen for a tilde, or the word "and" for "&"), instead of retyping the value.
    """
    if find == replace_with:
        return "no change: what you are replacing and what you replace it with are the same"
    if find not in text:
        return f"no change: {find!r} does not occur in the text"
    return text.replace(find, replace_with)


array_of_tools = [convert_units, remove_characters, replace_substring]
