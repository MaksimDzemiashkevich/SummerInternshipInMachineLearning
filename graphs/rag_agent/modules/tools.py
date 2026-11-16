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


def _format(magnitude: float, decimals: int) -> str:
    """Число в строку: с заданным количеством знаков либо без хвостовых нулей."""
    if decimals >= 0:
        return f"{magnitude:.{decimals}f}"
    text = f"{magnitude:.6f}".rstrip("0").rstrip(".")
    return text or "0"


_UNIT_NAME_HINT = ("give units as standard names or symbols: inch, ft, mm, m, kg, lb, t, psi, bar, "
                   "kPa, MPa, degC, degF, K, kW, hp, m**3/h, mm**2, rpm; give a gauge or absolute "
                   "pressure without its mark (barg -> bar)")


def _unit(name: str):
    """Единица по стандартному имени, которое дала модель.

    Строки значений код не разбирает — это делает модель (convert_units, pipeline/parsing.py).
    Здесь только то, чего нет в словаре pint: одинокие F и C в датащите — градусы, а не фарад и
    кулон, и инженерная запись степени («mm2», «m3/h», «m²»).
    """
    name = str(name).strip()
    name = {"F": "degF", "C": "degC", "°F": "degF", "°C": "degC"}.get(name, name)
    name = name.replace("²", "**2").replace("³", "**3")
    return _UREG.Unit(re.sub(r"(?<=[A-Za-z])([23])\b", r"**\1", name))


def unit_mismatch(left: str, right: str) -> bool:
    """Разные ли это единицы по существу. Не единицы (или нечитаемое) — не разные.

    Сравнивать написанием нельзя: «in.», «inch» и «in» — одно и то же. Сравнивает pint, он знает
    синонимы. Отметку манометрического/абсолютного давления сюда не передают: её отделяет разбор
    модели (parsing.units_differ).
    """
    try:
        return _unit(left) != _unit(right)
    except Exception:
        return False


def is_unit(word: str) -> bool:
    """Читается ли слово как единица измерения («in», «kPa», «rpm»), а не как содержимое записи."""
    try:
        return not _unit(word).dimensionless
    except Exception:
        return False


def same_dimension(left: str, right: str) -> bool:
    """Одна ли величина у двух единиц (стандартные имена, как их даёт разбор модели): lb и t — да."""
    try:
        return _unit(left).dimensionality == _unit(right).dimensionality
    except Exception:
        return False


def convert_value(value: float, from_unit: str, to_unit: str) -> float | None:
    """Точное число после пересчёта (для проверки ответа) или None, если пересчитать нельзя."""
    try:
        return float(_UREG.Quantity(float(value), _unit(from_unit)).to(_unit(to_unit)).magnitude)
    except Exception:
        return None


@tool
def convert_units(value: float, from_unit: str, to_unit: str, decimals: int = -1) -> str:
    """Convert a number from one unit to another, exactly.

    Read the recorded value yourself and pass it already taken apart: `value` is the number
    alone and `from_unit` the unit it is recorded in. For "0.5-inch" pass value=0.5,
    from_unit="inch"; for '2 1/2"' value=2.5, from_unit="inch"; for "7,62 cm" value=7.62,
    from_unit="cm"; for "1200 F" value=1200, from_unit="degF". A range or a list is converted
    one number per call.
    `to_unit` is the unit wanted. Units are standard names or symbols: inch, ft, mm, m, kg, lb,
    t, psi, bar, kPa, MPa, degC, degF, K, kW, hp, m**3/h, mm**2, rpm and so on. A gauge or
    absolute pressure is passed without its mark ("barg" -> "bar", "psia" -> "psi"); put the
    mark back when you write the answer, as the rules require it.
    `decimals` rounds the result to that many decimal places; leave it out for the exact value
    with trailing zeros dropped.

    Returns "<number> <to_unit>", or a line starting with "cannot convert" when a unit is not
    known or the two units measure different things — then do not invent a number: fix the
    unit name and call again, or treat the conversion as not possible.
    """
    try:
        source, target = _unit(from_unit), _unit(to_unit)
    except Exception as e:
        return f"cannot convert: unknown unit ({e}); {_UNIT_NAME_HINT}"
    try:
        converted = _UREG.Quantity(float(value), source).to(target)
    except pint.DimensionalityError as e:
        return f"cannot convert: {e}"
    except Exception as e:
        return f"cannot convert: {e}; {_UNIT_NAME_HINT}"
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
