"""Подготовка загруженной таблицы и контекст строки."""
import re

import pandas as pd


# Колонки, которые пайплайн читает как текст: из Excel в них приходит что угодно.
_TEXT_COLUMNS = ("Attribute", "Value")


_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _cell_text(value) -> str:
    """Ячейка Excel → строка: пропуск пустой, целое без «.0», дата без нулевого времени.

    Датащит почти никогда не состоит из одних строк: размер лежит числом, дата испытаний —
    датой, пустая ячейка — NaN. Всё это раньше доезжало до value.strip() и роняло запрос.
    """
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d") if value == value.normalize() else str(value)
    if isinstance(value, float) and float(value).is_integer():
        return str(int(value))
    # управляющие символы Excel не хранит, а модели ломают построчную подачу значения
    return _CONTROL_CHARS_RE.sub("", str(value))


def _unique_columns(columns) -> list[str]:
    """Повторяющиеся заголовки разводит: вторая «Attribute» становится «Attribute (2)».

    Две колонки с одним именем превращают frame[col] в таблицу вместо ячейки, и строка
    падает на первой же нормализации имени.
    """
    seen: dict[str, int] = {}
    result = []
    for column in columns:
        name = str(column)
        seen[name] = seen.get(name, 0) + 1
        result.append(name if seen[name] == 1 else f"{name} ({seen[name]})")
    return result


def prepare_frame(data_frame: pd.DataFrame) -> pd.DataFrame:
    """Загруженная таблица → таблица, с которой работает пайплайн.

    Заголовки делаются уникальными, а Attribute и Value — текстом. Прочие колонки (контекст)
    остаются как есть: они идут в выдачу в том виде, в каком их прислали.
    """
    frame = data_frame.copy()
    frame.columns = _unique_columns(frame.columns)
    for column in _TEXT_COLUMNS:
        if column in frame.columns:
            frame[column] = [_cell_text(v) for v in frame[column]]
    return frame


def _row_context(data_frame: pd.DataFrame, index: int) -> str:
    """Контекст строки из колонок сверх Attribute/Value: «Item: pump body; Revision: 3».

    Заказчик может добавить в таблицу колонку, уточняющую, к чему относится строка (изделие,
    узел, ревизия). Такая колонка не нормализуется, но участвует в поиске правил и подаётся
    модели — без неё правила двух разных предметов с одинаковым именем атрибута неразличимы.
    """
    extra = [c for c in data_frame.columns if c not in ("Attribute", "Value")]
    parts = []
    for column in extra:
        cell = data_frame.iloc[index][column]
        if pd.notna(cell) and str(cell).strip():
            parts.append(f"{column}: {str(cell).strip()}")
    return "; ".join(parts)
