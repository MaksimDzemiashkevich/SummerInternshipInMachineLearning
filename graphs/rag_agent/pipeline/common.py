"""Общее для всего конвейера: логгер и сравнение имён атрибутов и фраз."""
import logging
import re

from dotenv import load_dotenv


load_dotenv()


logger = logging.getLogger("normalization_pipeline")


logger.setLevel(logging.INFO)


if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(handler)


def _context_line(context: str) -> str:
    """Строка контекста для промпта; пустой контекст не добавляет ничего."""
    return f"Row context: {context}\n" if context else ""


def _normalize_phrase(text: str) -> str:
    """Нормализует фразу для сравнения: нижний регистр, пунктуация→пробел, порядок слов не важен."""
    text = text.lower()
    text = re.sub(r"[()\-,]", " ", text)
    return " ".join(sorted(text.split()))


def _norm_attr(text: str) -> str:
    """Нормализует имя атрибута для сравнения (без сортировки слов, только регистр/пробелы)."""
    return re.sub(r"\s+", " ", text).strip().lower()


def _attribute_is_excluded(attribute: str, excluded: list[str]) -> bool:
    """Проверяет, попадает ли атрибут строки в список исключённых (по подстроке)."""
    a = _norm_attr(attribute)
    if not a or not excluded:
        return False
    for e in excluded:
        e_norm = _norm_attr(e)
        if not e_norm:
            continue
        if a in e_norm or e_norm in a:
            return True
    return False


def _attribute_matches(attribute: str, owner: str) -> bool:
    """Совпадает ли атрибут строки с атрибутом-владельцем правила.

    Сравнение по целым словам, а не по подстроке: «design» не должен считаться совпадением
    с «material designation», иначе к материалу подтягивается keyword совсем другой темы.
    """
    a = _norm_attr(attribute)
    o = _norm_attr(owner)
    if not a or not o:
        return True  # нет данных о владельце — применяем (backward compat)
    a_words = " " + " ".join(re.findall(r"\w+", a)) + " "
    o_words = " " + " ".join(re.findall(r"\w+", o)) + " "
    return o_words in a_words or a_words in o_words


def _names_clash(attribute: str, name: str) -> bool:
    """Разные ли это названные атрибуты: общее слово есть, но ни один не входит в другой.

    «nominal length» и «Nominal size» пересекаются словом «nominal» и при этом означают
    разное — шаблон одного к другому применять нельзя. А «dimension» и «Wear travel, in»
    не пересекаются вовсе: это общая группа, которой ретрив вправе накрыть строку.
    """
    a = set(re.findall(r"\w+", _norm_attr(attribute)))
    n = set(re.findall(r"\w+", _norm_attr(name)))
    if not a or not n or not (a & n):
        return False
    return not (a <= n or n <= a)


def _unit_key(text: str) -> str:
    """Ключ единицы измерения: без пробелов, точек и регистра ('  in.' == ' IN')."""
    return re.sub(r"[\s.]", "", text).lower()


# Единица не может начинаться с точки или запятой, иначе «9.5» разбирается как 9 + «.5».
_NUM_UNIT_RE = re.compile(r"\s*(-?\d+(?:[.,]\d+)?)(\s*)([^\d\s.,].*?)\s*")


_WORD_RE = re.compile(r"\w+")


def _digit_groups(text: str) -> set[str]:
    return set(re.findall(r"\d+", str(text)))


def _alnum(text: str) -> str:
    """Только буквы и цифры в нижнем регистре: «18Cr-10Ni» и «18Cr10Ni» дают одно и то же."""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def _char_bag(text: str) -> str:
    """Мультимножество букв и цифр: не зависит ни от порядка частей, ни от разделителей."""
    return "".join(sorted(re.findall(r"[a-z0-9]", str(text).lower())))
