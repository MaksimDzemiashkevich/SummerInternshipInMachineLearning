"""Нормализация таблицы по нескольким документам сразу и сборка итоговой таблицы."""
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from ..modules.config import AppConfig
from ..modules.chat import AdHocRule
from .common import logger
from .postprocess import _equivalence_key, _reading_signature
from .frames import _row_context, prepare_frame
from .graph import run_single_rule_set
from .user_rules import APPLIED_SUFFIX, USER_RULES_COLUMN, run_user_rules_column


def check_consistency(result_df: pd.DataFrame, rule_set_names: list[str],
                      contexts: list[str] | None = None) -> None:
    """Логирует warning, если к одинаковым по смыслу входам правила применены по-разному.

    Сравниваются не сами строки (две записи одного значения законно дают разные строки),
    а прочтение правил: заменено ли содержимое и какими разделителями собран результат.
    """
    contexts = contexts if contexts is not None else [""] * len(result_df)
    for col in rule_set_names:
        seen: dict[tuple, tuple[str, tuple]] = {}
        for attribute, value, result, context in zip(result_df["Attribute"], result_df["Value"],
                                                     result_df[col], contexts):
            key = _equivalence_key(attribute, value, context)
            signature = _reading_signature(value, result)
            if key in seen and seen[key][1] != signature:
                logger.warning(
                    "Inconsistent normalization for %r (col=%s): %r vs %r",
                    attribute, col, seen[key][0], result,
                )
            else:
                seen[key] = (result, signature)


def _free_column_name(frame: pd.DataFrame, name: str) -> str:
    """Имя для новой колонки, не затирающее уже существующую («STD-A» → «STD-A (2)»).

    Заказчик мог прислать таблицу, где такая колонка уже есть — например прошлый результат.
    Молча её перезаписать значит потерять данные пользователя.
    """
    if name not in frame.columns:
        return name
    suffix = 2
    while f"{name} ({suffix})" in frame.columns:
        suffix += 1
    return f"{name} ({suffix})"


def normalize_table(data_frame: pd.DataFrame, app_config: AppConfig,
                    rule_sets: list | None = None,
                    user_rules: list[AdHocRule] | None = None
                    ) -> tuple[pd.DataFrame, dict[str, list[bool]]]:
    """Прогоняет таблицу по выбранным документам; возвращает (таблицу, флаги спорных ячеек).

    Правила пользователя считаются отдельным проходом ПОСЛЕ стандартов и ложатся в свою
    колонку: колонки стандартов остаются чистыми — в них ровно то, что говорит документ.
    """
    rule_sets = list(app_config.rule_sets if rule_sets is None else rule_sets)
    data_frame = prepare_frame(data_frame)
    # в выдаче сохраняем все исходные колонки: уточняющие колонки нужны ревьюеру
    result_df = data_frame.copy()
    ambiguous_cols: dict[str, list[bool]] = {}
    # документы независимы друг от друга — считаем их колонки параллельно
    if rule_sets:
        with ThreadPoolExecutor(max_workers=len(rule_sets)) as pool:
            col_dfs = list(pool.map(
                lambda rs: run_single_rule_set(data_frame, app_config.model, rs),
                rule_sets,
            ))
        column_names = []
        for rule_set, col_df in zip(rule_sets, col_dfs):
            column = _free_column_name(result_df, rule_set.name)
            column_names.append(column)
            result_df[column] = col_df["Rules_Value"].to_list()
            if "Applied" in col_df:
                result_df[_free_column_name(result_df, column + APPLIED_SUFFIX)] = (
                    col_df["Applied"].to_list())
            if "Ambiguous" in col_df:
                ambiguous_cols[column] = col_df["Ambiguous"].to_list()

        check_consistency(result_df, column_names,
                          [_row_context(data_frame, i) for i in range(len(data_frame))])

    if user_rules:
        logger.info("Колонка %r: правил пользователя — %d", USER_RULES_COLUMN, len(user_rules))
        col_df = run_user_rules_column(data_frame, app_config.model, user_rules)
        column = _free_column_name(result_df, USER_RULES_COLUMN)
        result_df[column] = col_df["Rules_Value"].to_list()
        if "Applied" in col_df:
            result_df[_free_column_name(result_df, column + APPLIED_SUFFIX)] = (
                col_df["Applied"].to_list())
        ambiguous_cols[column] = col_df["Ambiguous"].to_list()

    return result_df, ambiguous_cols
