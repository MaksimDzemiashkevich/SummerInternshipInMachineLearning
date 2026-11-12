"""Нормализация таблицы по нескольким документам сразу и сборка итоговой таблицы."""
import os
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


# Чем считать колонку документа: «spec» — основным путём (spec_runner: паспорт атрибута, точные
# операции делает код, у каждого ответа уровень уверенности), «legacy» — запасным (граф по
# строкам). Документ без attribute_spec.json всегда считается запасным путём.
NORMALIZER = os.getenv("NORMALIZER", "spec").strip().lower()


def run_rule_set(data_frame: pd.DataFrame, app_config: AppConfig, rule_set) -> pd.DataFrame:
    """Колонка одного документа (ROW_COLUMNS) — основным путём или запасным."""
    return _run_rule_set(data_frame, app_config, rule_set)


def _run_rule_set(data_frame: pd.DataFrame, app_config: AppConfig, rule_set, progress=None) -> pd.DataFrame:
    """Колонка одного документа: основным путём (паспорта атрибутов), если у документа есть
    attribute_spec.json, иначе — запасным (граф по строкам, или всегда при NORMALIZER=legacy)."""
    from ..spec_runner import has_spec, run_spec_rule_set
    if NORMALIZER == "spec" and has_spec(rule_set):
        logger.info("Документ %r: нормализация по спецификации атрибутов", rule_set.name)
        # спецификации нужна модель без привязанных инструментов (структурированный ответ)
        return run_spec_rule_set(data_frame, app_config.spec_model or app_config.assistant or app_config.model,
                                 rule_set, progress)
    return run_single_rule_set(data_frame, app_config.model, rule_set, progress)


def normalize_table(data_frame: pd.DataFrame, app_config: AppConfig,
                    rule_sets: list | None = None,
                    user_rules: list[AdHocRule] | None = None,
                    progress=None) -> tuple[pd.DataFrame, dict[str, list[bool]]]:
    """Прогоняет таблицу по выбранным документам; возвращает (таблицу, флаги спорных ячеек).

    Правила пользователя считаются отдельным проходом ПОСЛЕ стандартов и ложатся в свою
    колонку: колонки стандартов остаются чистыми — в них ровно то, что говорит документ.
    progress (jobs.Progress) — счётчик готовых ячеек: строки × стандарты (+ колонка разовых правил).
    """
    rule_sets = list(app_config.rule_sets if rule_sets is None else rule_sets)
    data_frame = prepare_frame(data_frame)
    if progress is not None:
        progress.add_total(len(data_frame) * (len(rule_sets) + (1 if user_rules else 0)))
    # в выдаче сохраняем все исходные колонки: уточняющие колонки нужны ревьюеру
    result_df = data_frame.copy()
    ambiguous_cols: dict[str, list[bool]] = {}
    # разметка колонок результата — для вопросов по файлу и пересчёта отдельных строк
    source_columns = [str(c) for c in data_frame.columns]
    layout = {"source_columns": source_columns,
              "context_columns": [c for c in source_columns if c not in ("Attribute", "Value")],
              "standards": [], "user_rules": None}
    # документы независимы друг от друга — считаем их колонки параллельно
    if rule_sets:
        with ThreadPoolExecutor(max_workers=len(rule_sets)) as pool:
            col_dfs = list(pool.map(lambda rs: _run_rule_set(data_frame, app_config, rs, progress), rule_sets))
        column_names = []
        for rule_set, col_df in zip(rule_sets, col_dfs):
            column = _free_column_name(result_df, rule_set.name)
            column_names.append(column)
            result_df[column] = col_df["Rules_Value"].to_list()
            applied = _free_column_name(result_df, column + APPLIED_SUFFIX)
            result_df[applied] = (col_df["Applied"].to_list() if "Applied" in col_df
                                  else [""] * len(result_df))
            layout["standards"].append({"column": column, "applied": applied,
                                        "standard": rule_set.name})
            if "Ambiguous" in col_df:
                ambiguous_cols[column] = col_df["Ambiguous"].to_list()

        check_consistency(result_df, column_names,
                          [_row_context(data_frame, i) for i in range(len(data_frame))])

    if user_rules:
        logger.info("Колонка %r: правил пользователя — %d", USER_RULES_COLUMN, len(user_rules))
        col_df = run_user_rules_column(data_frame, app_config.model, user_rules)
        if progress is not None:
            progress.step(len(data_frame))
        column = _free_column_name(result_df, USER_RULES_COLUMN)
        result_df[column] = col_df["Rules_Value"].to_list()
        applied = _free_column_name(result_df, column + APPLIED_SUFFIX)
        result_df[applied] = col_df["Applied"].to_list()
        ambiguous_cols[column] = col_df["Ambiguous"].to_list()
        layout["user_rules"] = {"column": column, "applied": applied}

    result_df.attrs["layout"] = layout
    return result_df, ambiguous_cols
