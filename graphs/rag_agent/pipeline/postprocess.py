"""Пост-проходы по готовой колонке: согласование равнозначных строк, единое написание."""
import json
import re
from collections import Counter

from ..modules.system_prompts import RECONCILE_PROMPT
from .parsing import is_measure, parse, parse_many
from .common import _alnum, _char_bag, _digit_groups, _norm_attr, _normalize_phrase, logger
from .retrieval import build_rules_text
from .normalizer import _invoke_with_tools


def _strip_redundant_parenthetical(answer: str, original: str) -> str:
    """Убирает хвостовую скобку, которая лишь повторяет уже закодированное значение.

    Документ может предписывать «код (второе, отдельно записанное значение)». Если же в поле
    было одно значение, модель по инерции из worked-example дописывает его копию в скобках
    («A193 B7» → «XB193B7 (A193 B7)»). Признак избыточности: содержимое скобки не приносит
    ничего нового — его слова уже есть в исходном значении, а все числа исходного значения
    уже учтены в закодированной части.
    """
    stripped = str(answer).strip()
    inner_only = re.fullmatch(r"\(([^()]+)\)", stripped)
    if inner_only:
        # целиком скобка — это обрывок составной формы, а не самостоятельное значение
        logger.info("  [DEDUP] %r -> %r", stripped, inner_only.group(1).strip())
        return inner_only.group(1).strip()
    if _normalize_phrase(answer) == _normalize_phrase(original):
        return answer  # значение воспроизведено как есть — удалять нечего
    # скобка, прижатая к слову («bar(g)», «dB(A)»), — часть обозначения единицы, а не
    # приписанное значение: раньше «13.0 bar(g)» так обрезалось до «13.0 bar»
    m = re.fullmatch(r"([^()]+?)\s+\(([^()]+)\)", str(answer).strip())
    if not m:
        return answer
    head, inner = m.group(1).strip(), m.group(2).strip()
    if not head:
        return answer
    original_words = set(_normalize_phrase(original).split())
    inner_words = set(_normalize_phrase(inner).split())
    if not inner_words or not inner_words <= original_words:
        return answer  # в скобке что-то своё — это не повтор
    if not _digit_groups(original) <= _digit_groups(head):
        return answer  # часть значения закодирована только в скобке — скобка нужна
    logger.info("  [DEDUP] %r -> %r", answer, head)
    return head


def _equivalence_key(attribute: str, value: str, context: str = "") -> tuple:
    """Ключ «одно и то же значение, записанное по-другому».

    Игнорирует порядок частей и порядок символов внутри части, поэтому «C-Mn-Si» и «Si-Mn-C»,
    «18Cr-10Ni-Mo» и «Cr18-Ni10-Mo» попадают в один класс: правила к ним должны быть
    прочитаны одинаково, даже если итоговые строки останутся разными.
    """
    parts = sorted("".join(sorted(p)) for p in re.findall(r"[a-z0-9]+", str(value).lower()))
    return _norm_attr(attribute), _norm_attr(context), tuple(parts)


def _reading_signature(value: str, answer: str) -> tuple:
    """Как именно правила были прочитаны для строки, независимо от её собственного написания.

    Две записи одного и того же значения обязаны дать один и тот же ответ на вопросы «содержимое
    оставлено как есть или заменено?» и «какие разделители в итоге стоят?» — но сами строки при
    этом законно остаются разными («C-Mn-Si» и «Si-Mn-C», «DN 150» и «150 DN»), поэтому порядок
    частей в сравнении не участвует.
    """
    kept = _char_bag(answer) == _char_bag(value)
    if kept:
        return True, tuple(sorted(re.findall(r"[^A-Za-z0-9]", str(answer))))
    return False, _char_bag(answer)


def _reconcile_equivalent_rows(rows: list[dict], rule_set, model) -> list[dict]:
    """Сверяет строки, которые по смыслу означают одно и то же, но нормализовались по-разному.

    Одинаковое по сути значение, записанное в другом порядке слов, должно трактоваться одним
    и тем же прочтением правил. Если прочтения разошлись, модель один раз выбирает общее — и
    применяет его к каждой записи отдельно (порядок слов и написание каждой записи свои).
    Детерминированные и спорные (pending/ambiguous) строки не трогаем.
    """
    if model is None:
        return rows

    classes: dict[tuple[str, str], list[int]] = {}
    for i, row in enumerate(rows):
        if row.get("Deterministic") or row.get("Ambiguous"):
            continue
        key = _equivalence_key(row["Attribute"], row["Value"], row.get("Context", ""))
        classes.setdefault(key, []).append(i)

    for indices in classes.values():
        if len(indices) < 2:
            continue
        if len({_reading_signature(rows[i]["Value"], rows[i]["Rules_Value"]) for i in indices}) < 2:
            continue  # прочтение одно и то же; сами строки и должны отличаться
        attribute = rows[indices[0]]["Attribute"]
        context = str(rows[indices[0]].get("Context", ""))
        rules_text, _names, _conflict = build_rules_text(
            rule_set, attribute, str(rows[indices[0]]["Value"]), context)
        listing = "\n".join(
            f"{n}. recorded: {rows[i]['Value']}  ->  current answer: {rows[i]['Rules_Value']}"
            for n, i in enumerate(indices, start=1)
        )
        user = (
            f"Attribute: {attribute}\n\n"
            f"These recorded values denote the same thing but were normalized inconsistently:\n"
            f"{listing}\n\n"
            f"Rules:\n{rules_text}"
        )
        messages = [{"role": "system", "content": RECONCILE_PROMPT},
                    {"role": "user", "content": user}]
        try:
            raw = str(_invoke_with_tools(model, messages)[0].content).strip()
            raw = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", raw).strip()
            decided = json.loads(raw)
        except Exception as e:
            logger.warning("  [RECONCILE] failed: %s", e)
            continue
        for n, i in enumerate(indices, start=1):
            new_value = decided.get(str(n))
            if not isinstance(new_value, str) or not new_value.strip():
                continue
            new_value = new_value.strip()
            current = str(rows[i]["Rules_Value"]).strip()
            if new_value == current:
                continue
            # сверка вправе поменять только прочтение правил (разделители, регистр, пробелы)
            # либо вернуть запись как есть — но не переставлять и не подменять её содержимое
            if _alnum(new_value) not in (_alnum(rows[i]["Value"]), _alnum(current)):
                logger.warning("  [RECONCILE] отклонено (меняет содержимое): %r -> %r",
                               current, new_value)
                continue
            logger.info("  [RECONCILE] %r: %r -> %r", rows[i]["Value"], current, new_value)
            _note_row(rows[i], "сверка одинаковых по смыслу записей: прочтение правил "
                               "приведено к общему")
            rows[i]["Rules_Value"] = new_value
    return rows


def _note_row(row: dict, note: str) -> None:
    """Дописывает строку объяснения: проходы по всей таблице идут после сборки заметок."""
    current = str(row.get("Applied", "")).strip()
    empty = ("правила этого документа к строке не применялись", "")
    row["Applied"] = note if current in empty else f"{current}; {note}"


def _canonicalize_formatting(rows: list[dict]) -> list[dict]:
    """Приводит к одному написанию результаты, различающиеся ТОЛЬКО пробелами и точкой единицы.

    Одно и то же значение, записанное во входе по-разному ("1200 F" / "1200°F"), не должно
    давать "649 °C" и "649°C" в одной колонке; так же не должны расходиться "3.000 in" и
    "11.811 in." — документ пишет единицу с точкой, модель ставит её через раз. Сливаются
    только варианты, совпадающие после удаления пробелов и завершающей точки: смысловые
    различия (и точка внутри числа) не затрагиваются.
    """
    def key_of(value: str) -> str:
        return re.sub(r"\.$", "", re.sub(r"\s+", "", str(value)))

    buckets: dict[tuple[str, str], list[int]] = {}
    for i, row in enumerate(rows):
        key = (_norm_attr(row["Attribute"]), _norm_attr(row.get("Context", "")),
               key_of(row["Rules_Value"]))
        buckets.setdefault(key, []).append(i)

    for indices in buckets.values():
        variants = [str(rows[i]["Rules_Value"]) for i in indices]
        if len(set(variants)) < 2:
            continue
        counts = Counter(variants)
        (canonical, top), *rest = counts.most_common()
        if rest and rest[0][1] == top:
            # ничья: какой из вариантов требует документ, отсюда не видно — не выбираем за него
            logger.info("  [CANON] %r: большинства нет, оставляем как есть", sorted(set(variants)))
            continue
        logger.info("  [CANON] %r -> %r", sorted(set(variants)), canonical)
        for i in indices:
            if str(rows[i]["Rules_Value"]) != canonical:
                _note_row(rows[i], f"написание сведено к варианту «{canonical}» "
                                   f"(так нормализованы такие же записи этой таблицы)")
            rows[i]["Rules_Value"] = canonical
    return rows


def _canonicalize_units(rows: list[dict]) -> list[dict]:
    """Одна единица — одно написание в пределах атрибута: «3.000 in» и «11.811 in.» вместе.

    Значения разные, поэтому предыдущий проход их не сливает, а документ пишет единицу одним
    способом («in.»), модель же ставит точку через раз. Выбирается самое частое написание, при
    равенстве — ничего: сокращать то, что документ написал с точкой, мы не вправе. Число,
    единицу и её написание даёт разбор модели; одна ли это единица, решает pint.
    """
    parse_many([(str(row["Rules_Value"]), "") for row in rows])
    parts: dict[int, tuple] = {}
    groups: dict[tuple, list[int]] = {}
    for i, row in enumerate(rows):
        answer = str(row["Rules_Value"]).strip()
        parsed = parse(answer)
        q = parsed.single if parsed else None
        if (q is None or not q.unit_text or not is_measure(q.unit)
                or f"{q.number}{' ' if q.space else ''}{q.unit_text}" != answer):
            continue  # не одно число с единицей — сводить написание нечего
        parts[i] = (q.number, " " if q.space else "", q.unit_text)
        key = (_norm_attr(row["Attribute"]), _norm_attr(row.get("Context", "")), q.unit, q.reference)
        groups.setdefault(key, []).append(i)

    for indices in groups.values():
        spellings = [parts[i][2] for i in indices]
        if len(set(spellings)) < 2:
            continue
        counts = Counter(spellings)
        (canonical, top), *rest = counts.most_common()
        if rest and rest[0][1] == top:
            # ничья: «Pa» и «PA» — разные единицы, выбирать за документ мы не вправе
            logger.info("  [UNIT] %r: большинства нет, оставляем как есть", sorted(set(spellings)))
            continue
        logger.info("  [UNIT] %r -> %r", sorted(set(spellings)), canonical)
        for i in indices:
            number, space, _ = parts[i]
            fixed = f"{number}{space}{canonical}"
            if fixed != str(rows[i]["Rules_Value"]):
                _note_row(rows[i], f"написание единицы сведено к «{canonical.strip()}» "
                                   f"(как в остальных строках этого атрибута)")
            rows[i]["Rules_Value"] = fixed
    return rows


