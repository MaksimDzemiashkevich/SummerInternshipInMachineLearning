"""Колонка правил пользователя: какие правила относятся к строке и их применение."""
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from ..modules.chat import AdHocRule, format_adhoc_rules
from ..modules.tools import same_quantity
from .common import _attribute_matches, logger
from .answers import _is_ambiguous_answer
from .guards import _partial_separator_edit, _records_quantity
from .postprocess import _canonicalize_formatting
from .frames import _row_context, prepare_frame
from .normalizer import SELF_CONSISTENCY, _vote_normalize
from .graph import ROW_COLUMNS


def _adhoc_rules_for_attribute(rules: list[AdHocRule], attribute: str, value: str = "",
                               model=None, context: str = "") -> list[AdHocRule]:
    """Правила пользователя, относящиеся к этой строке: общие + названные для её атрибута.

    Имя атрибута пользователь пишет своими словами («материал» против «Material designation»),
    поэтому сравнение то же пословное, что и для владельцев правил документа. Правило о роде
    величины («всю массу — в тонны») решается не по имени, а по самому значению: масса
    называлась «Weight», правило пришло с атрибутом «Mass» и не нашло ни одной строки. Единица
    значения ещё не значит величину, поэтому модель (если есть) подтверждает, что атрибут —
    величина этого рода, а не класс, записанный «в фунтах».
    """
    def applies(rule: AdHocRule) -> bool:
        if rule.quantity:
            return (same_quantity(value, rule.quantity, attribute)
                    and (model is None
                         or _records_quantity(model, attribute, value, rule.quantity, context)))
        return not rule.attribute or _attribute_matches(attribute, rule.attribute)

    return [r for r in rules if applies(r)]


def user_rule_hits(data_frame: pd.DataFrame, rules: list[AdHocRule], model=None) -> list[int]:
    """Скольким строкам таблицы подходит каждое правило пользователя (в порядке правил)."""
    frame = prepare_frame(data_frame)
    hits = [0] * len(rules)
    for index in range(frame.shape[0]):
        matched = _adhoc_rules_for_attribute(rules, frame.iloc[index]["Attribute"],
                                             frame.iloc[index]["Value"], model,
                                             _row_context(frame, index))
        for i, rule in enumerate(rules):
            hits[i] += rule in matched
    return hits


# Колонка под правила, названные пользователем в сообщении.
USER_RULES_COLUMN = "Мои правила"


# Суффикс колонки-объяснения: что именно сработало на этой строке.
APPLIED_SUFFIX = " — применено"


# Строки этого прохода независимы друг от друга, поэтому считаются параллельно; ширина
# подобрана так, чтобы вместе с сэмплами не устраивать шквал запросов к модели.
USER_RULES_WORKERS = 4


def run_user_rules_column(data_frame: pd.DataFrame, model,
                          rules: list[AdHocRule],
                          samples: int = SELF_CONSISTENCY) -> pd.DataFrame:
    """Отдельная колонка под правила пользователя: они применяются к самой записи.

    Стандарты в этой колонке не участвуют — по ним считаются свои колонки, и правило
    пользователя их не трогает. Строка, к атрибуту которой правил нет, остаётся как
    записана: колонка показывает ровно то, что делают правила пользователя, и ничего сверх.
    """
    samples = max(1, int(samples))

    def one_row(index: int) -> dict:
        attribute = data_frame.iloc[index]["Attribute"]
        value = data_frame.iloc[index]["Value"]
        context = _row_context(data_frame, index)
        row = {"Attribute": attribute, "Value": value, "Rules_Value": value,
               "Ambiguous": False, "Context": context, "Deterministic": False, "Applied": ""}

        row_rules = _adhoc_rules_for_attribute(rules, attribute, value, model, context)
        if not row_rules:
            return row

        answer, split, applied = _vote_normalize(model, attribute, value,
                                                 format_adhoc_rules(row_rules), context, samples)
        row["Applied"] = "; ".join(
            [f"ваше правило: {r.text}" for r in row_rules] + ([applied] if applied else []))
        logger.info("  [USER RULES] %r | %r -> %r", attribute, value, answer)
        if _is_ambiguous_answer(answer):
            # правила пользователя противоречат друг другу на этой строке — решать не нам
            row["Ambiguous"] = True
            return row

        row["Rules_Value"] = answer
        # разошедшиеся сэмплы и правка разделителей не по всем вхождениям — к ревьюеру
        partial = _partial_separator_edit(value, answer)
        if partial:
            logger.info("  [USER RULES] %r -> %r: правка не по всем вхождениям (%s)",
                        value, answer, partial)
        row["Ambiguous"] = split or bool(partial)
        return row

    with ThreadPoolExecutor(max_workers=USER_RULES_WORKERS) as pool:
        rows = list(pool.map(one_row, range(data_frame.shape[0])))
    return pd.DataFrame(_canonicalize_formatting(rows), columns=ROW_COLUMNS)
