# -*- coding: utf-8 -*-
"""Запись числа по шаблону ICU/CLDR (Babel) — общая для компиляции спецификации и её исполнения.

Модель пишет в спецификацию стандартный шаблон числа ("0.0" — ровно один знак, "0.#" — до одного
знака без нулей в конце, "#,##0.00", "@@@") и локаль, чьи знаки разделителей велит документ; число
пишет Babel. Прежде были свои поля decimals / max_decimals / decimal_separator, и модель путала
«показывать N знаков» с «округлять до N знаков» («156 kg» → «156.0 kg»): в шаблоне это два разных
написания, а не толкование одного поля.
"""
from __future__ import annotations

from decimal import ROUND_DOWN, ROUND_HALF_EVEN, ROUND_HALF_UP, ROUND_UP, Decimal, localcontext

from babel import Locale
from babel.numbers import format_decimal, parse_pattern

ROUNDING = {"half_up": ROUND_HALF_UP, "half_even": ROUND_HALF_EVEN, "down": ROUND_DOWN, "up": ROUND_UP}


def valid_number_pattern(pattern) -> bool:
    if not isinstance(pattern, str):
        return False
    if not pattern:
        return True
    if set(pattern) - set("0#@.,") or not set(pattern) & set("0#@"):
        return False  # единица, текст или условие в шаблоне числа — их код не исполняет
    whole, _, fraction = pattern.partition(".")
    if "," in fraction or any(len(group) != 3 for group in whole.split(",")[1:]):
        # в шаблоне ICU запятая — разделитель тысяч (группа из трёх цифр), а не десятичная
        # запятая документа: ту задаёт locale. «0,00» Babel пишет как «0 03».
        return False
    try:
        parse_pattern(pattern)
        return True
    except Exception:
        return False


def valid_locale(locale) -> bool:
    try:
        Locale.parse(str(locale).replace("-", "_"))
        return True
    except Exception:
        return False


def format_number(number: float, pattern: str, locale: str = "en", rounding: str = "half_up") -> str:
    """Число по шаблону ICU в знаках локали; округление — как велит документ (по умолчанию 0.5 вверх)."""
    with localcontext() as context:
        context.rounding = ROUNDING.get(rounding or "half_up", ROUND_HALF_UP)
        context.prec = 100  # шаблон с десятками «#» глубже точности по умолчанию (28) — InvalidOperation
        return format_decimal(Decimal(str(number)), format=pattern,
                              locale=str(locale or "en").replace("-", "_"))


def keep_digits(pattern: str, number: float) -> str:
    """Тот же шаблон, но без округления: знаков после разделителя не меньше, чем у самого числа.
    Для значения, которое документ велит не округлять («округляются только пересчитанные»);
    обязательные нули шаблона и разделитель тысяч остаются."""
    if "@" in pattern:
        return pattern
    whole, _, fraction = pattern.partition(".")
    digits = max(0, -Decimal(str(number)).normalize().as_tuple().exponent)
    zeros = fraction.count("0")
    width = max(len(fraction), digits)
    return whole + ("." + "0" * zeros + "#" * (width - zeros) if width else "")
