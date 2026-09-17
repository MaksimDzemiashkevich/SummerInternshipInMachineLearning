import io
import os
import re
import json
import logging
from rapidfuzz import fuzz
from functools import lru_cache
from typing import Literal, Annotated
from operator import add
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from dotenv import load_dotenv

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Depends, UploadFile, File
from fastapi.responses import Response

from langgraph.graph import START, StateGraph, MessagesState, END
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import RemoveMessage, ToolMessage

from .modules.config import AppConfig, Settings, build_app_config, hybrid_search, _extract_code_prefix
from .modules.system_prompts import (SYSTEM_PROMPT1, VERIFY_PROMPT, REGISTER_PROMPT,
                                     RECONCILE_PROMPT)
from .modules.tools import array_of_tools

TOOLS_BY_NAME = {t.name: t for t in array_of_tools}

load_dotenv()

logger = logging.getLogger("normalization_pipeline")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(handler)

RETRIEVAL_K = 4
# Сколько независимых сэмплов усредняется голосованием на LLM-пути (1 — выключено).
# Больше сэмплов — устойчивее ответ на спорных строках, но дороже прогон.
SELF_CONSISTENCY = int(os.getenv("NORMALIZER_SAMPLES", "3"))


class GraphState(MessagesState):
    index_row_table: int
    attribute: str
    value: str
    original_value: str
    deterministic_result: str | None
    answer: str | None
    needs_review_flag: bool
    new_rows: Annotated[list[dict], add]
    data_frame: pd.DataFrame
    rules_text: str
    rule_group_names: list[str]
    row_context: str


def build_rules_text(rule_set, attribute: str, value: str, context: str = "",
                     top_k: int = RETRIEVAL_K) -> tuple[str, list[str]]:
    """Полный набор правил для одной строки: global + релевантные local-группы.

    Ретрив видит и атрибут, и само значение (в значении может быть признак исключения —
    префикс, скобка, диапазон — из-за которого релевантна другая группа правил).
    Возвращает (текст правил, имена найденных групп).
    """
    blocks: list[str] = []
    if rule_set.global_rules:
        blocks.append(rule_set.global_rules)

    query = " ".join(x for x in (attribute, str(value), context) if x).strip()
    body_to_name = {g.body: g.attribute for g in rule_set.groups}
    names: list[str] = []
    for idx, body in enumerate(hybrid_search(rule_set, query, top_k=top_k)):
        name = body_to_name.get(body, "")
        logger.info("  [SEARCH] Rule %d: %r | %r", idx + 1, name, body[:100])
        if name:
            names.append(name)
        if body not in blocks:
            blocks.append(body)

    if not blocks:
        return "No relevant rules were found in the knowledge base for this attribute.", names

    text = "\n".join(blocks)
    return "\n\n".join([text] + expand_references(rule_set, text, attribute)), names


_CLAUSE_DEF_RE = re.compile(r"^\s*(\d+(?:\.\d+)+)\s")
_CLAUSE_REF_RE = re.compile(r"\b\d+(?:\.\d+)+\b")
# Упоминание таблицы/приложения/реестра — признак, что правило отсылает к списку значений.
_TABLE_REF_RE = re.compile(r"\b(?:table|appendix|register|annex)\b", re.I)
MAX_PULLED_CLAUSES = 12


def _context_line(context: str) -> str:
    """Строка контекста для промпта; пустой контекст не добавляет ничего."""
    return f"Row context: {context}\n" if context else ""


def _register_lines_for_attribute(rule_set, attribute: str) -> str:
    """Строки таблиц документа, относящиеся к этому атрибуту: keyword→root и фраза→код.

    Правило говорит «взять корень из Table 1», но самой таблицы в тексте правил нет — она
    лежит отдельным артефактом извлечения. Без неё модели нечего подставить, и она берёт
    первое похожее значение из примеров соседнего атрибута.
    """
    lines: list[str] = []
    for keyword, root in rule_set.deterministic.items():
        owner = rule_set.deterministic_attributes.get(keyword, "")
        if owner and _attribute_matches(attribute, owner):
            lines.append(f"{keyword} -> {root}")
    for prefix in rule_set.attribute_to_register.get(attribute.strip().lower(), []):
        for phrase, code in rule_set.closed_registers.get(prefix, {}).items():
            lines.append(f"{phrase} -> {code}")
    return "\n".join(dict.fromkeys(lines))


def expand_references(rule_set, text: str, attribute: str) -> list[str]:
    """Дотягивает то, на что найденные правила ссылаются: пункты по номеру и таблицы атрибута.

    Ссылка разрешается только если её номер реально определён в документе, поэтому десятичные
    числа в значениях («0.25 in») за ссылку не принимаются. Глубина — два уровня, с потолком
    по количеству, чтобы контекст строки не разрастался.
    """
    blocks: list[str] = []

    present = {m.group(1) for line in text.splitlines() if (m := _CLAUSE_DEF_RE.match(line))}
    pending = [r for r in dict.fromkeys(_CLAUSE_REF_RE.findall(text))
               if r in rule_set.clause_index and r not in present]
    pulled: list[str] = []
    while pending and len(pulled) < MAX_PULLED_CLAUSES:
        number = pending.pop(0)
        if number in present:
            continue
        present.add(number)
        line = rule_set.clause_index[number]
        pulled.append(line)
        for nested in _CLAUSE_REF_RE.findall(line):  # ссылки второго уровня
            if nested in rule_set.clause_index and nested not in present:
                pending.append(nested)
    if pulled:
        logger.info("  [REFS] подтянуто пунктов по ссылкам: %d", len(pulled))
        blocks.append("Clauses referenced above, quoted from the same document:\n"
                      + "\n".join(pulled))

    if _TABLE_REF_RE.search(text):
        register = _register_lines_for_attribute(rule_set, attribute)
        if register:
            logger.info("  [REFS] подтянута таблица атрибута: %d строк",
                        len(register.splitlines()))
            blocks.append(
                "Register entries this document defines for this attribute. This list is "
                "complete and closed: use an entry from it, and never invent one that is "
                "not listed.\n" + register)
    return blocks


def _is_ambiguous_answer(answer) -> bool:
    """Whether the model flagged the row as ambiguous (the AMBIGUOUS token)."""
    text = str(answer).strip()
    # Strip markdown code fences, quotes and trailing punctuation models often add.
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    text = text.strip('`"\'*_#,.()[]:; ')
    return text.upper() == "AMBIGUOUS"


def _is_pending_result(answer) -> bool:
    """True, если детерминированный исход — pending/ambiguous (требует ручного решения)."""
    text = str(answer).lower()
    return PENDING_WORDING in text or AMBIGUOUS_WORDING in text


def _parse_result(answer) -> str:
    """Извлекает итоговое значение из структурированного ответа модели (fallback на сырой текст)."""
    text = str(answer).strip()
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "result" in obj:
            return str(obj["result"])
    except Exception:
        pass
    return text


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
    """Совпадает ли атрибут строки с атрибутом-владельцем keyword (по подстроке)."""
    a = _norm_attr(attribute)
    o = _norm_attr(owner)
    if not a or not o:
        return True  # нет данных о владельце — применяем (backward compat)
    return o in a or a in o


def _match_render_template(attribute: str, render_templates: dict,
                           group_names: list[str] | None = None) -> dict | None:
    """Находит шаблон рендера: сперва по группам, которые ретрив счёл релевантными для строки,
    затем по имени самого атрибута (имя группы + алиасы, подстрока).

    Имя атрибута в датащите может не иметь ничего общего с именем группы правил
    («Wear travel, in» ↔ группа «dimension»), поэтому ретрив здесь — основной путь.
    """
    for name in group_names or []:
        tmpl = render_templates.get(name) or render_templates.get(_norm_attr(name))
        if tmpl:
            return tmpl
    a = _norm_attr(attribute)
    if not a:
        return None
    for attr, tmpl in render_templates.items():
        for c in [attr] + list(tmpl.get("aliases", [])):
            c_norm = _norm_attr(c)
            if c_norm and (c_norm in a or a in c_norm):
                return tmpl
    return None


def _unit_key(text: str) -> str:
    """Ключ единицы измерения: без пробелов, точек и регистра ('  in.' == ' IN')."""
    return re.sub(r"[\s.]", "", text).lower()


def _apply_render_template(answer: str, template: dict) -> str:
    """Детерминированно переформатирует ответ под шаблон (decimals + suffix).

    Шаблон только ПЕРЕрисовывает уже полученное значение и никогда не навешивает единицу:
    применяется, лишь если ответ — одно число со «своим» суффиксом, совпадающим с шаблонным
    с точностью до пробелов/точек. Иначе (диапазон, код, другая единица) ответ не трогаем.
    """
    m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)(.*)", str(answer).strip())
    if not m:
        return answer
    number, tail = m.group(1), m.group(2)
    if _unit_key(tail) != _unit_key(template["suffix"]):
        return answer
    return f"{float(number):.{template['decimals']}f}" + template["suffix"]


def _digit_groups(text: str) -> set[str]:
    return set(re.findall(r"\d+", str(text)))


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
    m = re.fullmatch(r"([^()]+?)\s*\(([^()]+)\)", str(answer).strip())
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


def _alnum(text: str) -> str:
    """Только буквы и цифры в нижнем регистре: «18Cr-10Ni» и «18Cr10Ni» дают одно и то же."""
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def _equivalence_key(attribute: str, value: str, context: str = "") -> tuple:
    """Ключ «одно и то же значение, записанное по-другому».

    Игнорирует порядок частей и порядок символов внутри части, поэтому «C-Mn-Si» и «Si-Mn-C»,
    «18Cr-10Ni-Mo» и «Cr18-Ni10-Mo» попадают в один класс: правила к ним должны быть
    прочитаны одинаково, даже если итоговые строки останутся разными.
    """
    parts = sorted("".join(sorted(p)) for p in re.findall(r"[a-z0-9]+", str(value).lower()))
    return _norm_attr(attribute), _norm_attr(context), tuple(parts)


def _char_bag(text: str) -> str:
    """Мультимножество букв и цифр: не зависит ни от порядка частей, ни от разделителей."""
    return "".join(sorted(re.findall(r"[a-z0-9]", str(text).lower())))


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
        rules_text, _ = build_rules_text(rule_set, attribute,
                                         str(rows[indices[0]]["Value"]), context)
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
            raw = str(_invoke_with_tools(model, messages).content).strip()
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
            rows[i]["Rules_Value"] = new_value
    return rows


def _canonicalize_formatting(rows: list[dict]) -> list[dict]:
    """Приводит к одному написанию результаты, различающиеся ТОЛЬКО пробелами.

    Одно и то же значение, записанное во входе по-разному ("1200 F" / "1200°F"), не должно
    давать "649 °C" и "649°C" в одной колонке. Сливаются только варианты, совпадающие после
    удаления пробелов, — смысловые различия не затрагиваются.
    """
    buckets: dict[tuple[str, str], list[int]] = {}
    for i, row in enumerate(rows):
        key = (_norm_attr(row["Attribute"]), _norm_attr(row.get("Context", "")),
               re.sub(r"\s+", "", str(row["Rules_Value"])))
        buckets.setdefault(key, []).append(i)

    for indices in buckets.values():
        variants = [str(rows[i]["Rules_Value"]) for i in indices]
        if len(set(variants)) < 2:
            continue
        # самый частый вариант; при равенстве — самый компактный, затем лексикографически
        canonical = sorted(set(variants), key=lambda v: (-variants.count(v), len(v), v))[0]
        logger.info("  [CANON] %r -> %r", sorted(set(variants)), canonical)
        for i in indices:
            rows[i]["Rules_Value"] = canonical
    return rows


FUZZY_THRESHOLD = 90
# Порог «похоже хоть на что-то»: ниже него реестр даже не обсуждается с моделью.
REGISTER_FLOOR = 40
# Формулировки исходов (легко локализовать/поменять)
PENDING_WORDING = "pending registration"
AMBIGUOUS_WORDING = "ambiguous match"


def _prefix_of(table: dict[str, str]) -> str:
    """Буквенная часть первого кода таблицы = префикс регистра."""
    return _extract_code_prefix(next(iter(table.values()))) or "REG"


def _adjudicate_register(model, attribute: str, value: str, table: dict[str, str]) -> str | None:
    """Спрашивает модель, какому пункту закрытого регистра соответствует значение.

    Возвращает код, "AMBIGUOUS", "UNMATCHED" или None (модель недоступна/ответ не разобран).
    Fuzzy-метрика видит только буквы: переформулировку («Gate, solid wedge» и «One-piece wedge,
    solid») она путает с другим пунктом реестра, поэтому серую зону разбирает модель — но
    выбирать ей разрешено только из самого реестра.
    """
    if model is None:
        return None
    register = "\n".join(f"{code} = {phrase}" for phrase, code in table.items())
    messages = [
        {"role": "system", "content": REGISTER_PROMPT},
        {"role": "user", "content": (f"Attribute: {attribute}\nRecorded value: {value}\n\n"
                                     f"Register:\n{register}")},
    ]
    try:
        raw = str(model.invoke(messages).content).strip()
    except Exception as e:  # сеть/квота — не роняем весь прогон, откатываемся на fuzzy
        logger.warning("  [REGISTER] adjudication failed: %s", e)
        return None
    verdict = raw.strip().strip('`"\'*_#.,;: ').splitlines()[0].strip() if raw else ""
    codes = {c.lower(): c for c in table.values()}
    if verdict.lower() in codes:
        return codes[verdict.lower()]
    if verdict.upper().startswith("AMBIG"):
        return "AMBIGUOUS"
    if verdict.upper().startswith("UNMATCH"):
        return "UNMATCHED"
    logger.warning("  [REGISTER] unparsed verdict %r", raw[:80])
    return None


def _content_words(text: str) -> tuple[str, ...]:
    """Слова значения без пунктуации и скобок; порядок слов не важен, состав — важен."""
    return tuple(sorted(re.findall(r"[a-z0-9]+", str(text).lower())))


def _without_parentheses(text: str) -> str:
    return re.sub(r"\([^()]*\)", " ", str(text))


def _wording_matches(value: str, phrase: str) -> bool:
    """Совпадает ли запись с пунктом реестра по СОСТАВУ слов.

    Запятые, дефисы, порядок слов и скобочное уточнение («Threaded end (NPT)») различаться
    могут: это то же самое описание. Лишнее или недостающее значащее слово — уже нет: код в
    таком случае присуждается «на глазок» и должен попасть к ревьюеру, а не быть принят молча.
    """
    value_variants = {_content_words(value), _content_words(_without_parentheses(value))}
    phrase_variants = {_content_words(phrase), _content_words(_without_parentheses(phrase))}
    return bool(value_variants & phrase_variants)


def match_closed_register(value: str, table: dict[str, str], threshold: int = FUZZY_THRESHOLD,
                          prefix: str | None = None, model=None,
                          attribute: str = "") -> tuple[str, bool] | None:
    """Сопоставляет значение с одним закрытым регистром «фраза таблицы -> код».

    Возвращает (результат, совпадение точное по составу слов). Точное/почти точное совпадение
    → код; серая зона отдаётся модели-арбитру (выбор только из реестра); «не похоже ни на что»
    → pending registration; «похоже сразу на два пункта» → ambiguous match. Префикс регистра
    берётся из самих кодов, не из среза строки.
    """
    if not table:
        return None
    norm_value = _normalize_phrase(value)
    scored = sorted(
        ((fuzz.token_sort_ratio(norm_value, _normalize_phrase(key)), key) for key in table),
        reverse=True,
    )
    best_score, best_key = scored[0]
    if best_score >= threshold:
        if _wording_matches(value, best_key):
            return table[best_key], True
        # похоже на пункт реестра, но формулировка другая — записываем как есть, не кодируем
        logger.info("  [REGISTER] %r ~ %r, формулировка не совпадает: оставляем как записано",
                    value, best_key)
        return value.strip(), False

    # Префикс берём от реально ближайшего кандидата, а не от произвольного/первого в списке.
    best_prefix = _extract_code_prefix(table[best_key]) or prefix or _prefix_of(table)

    if best_score >= REGISTER_FLOOR:
        verdict = _adjudicate_register(model, attribute, value, table)
        if verdict and verdict not in ("AMBIGUOUS", "UNMATCHED"):
            # код присуждается, только если у него есть пункт с тем же составом слов;
            # «по смыслу похоже» — это приближение, а приближать закрытый реестр нельзя:
            # запись остаётся как есть и уходит к ревьюеру
            if any(code == verdict and _wording_matches(value, phrase)
                   for phrase, code in table.items()):
                return verdict, True
            logger.info("  [REGISTER] %r ~ %r по смыслу, формулировка не совпадает: "
                        "оставляем как записано", value, verdict)
            return value.strip(), False
        if verdict == "AMBIGUOUS":
            return f"{best_prefix} {AMBIGUOUS_WORDING}: {value.strip()}", False
        if verdict == "UNMATCHED":
            return f"{best_prefix} {PENDING_WORDING}: {value.strip()}", False
    return f"{best_prefix} {PENDING_WORDING}: {value.strip()}", False


def apply_deterministic(value: str, rule_set, attribute: str,
                        model=None) -> tuple[str, bool] | None:
    """Детерминированные преобразования.

    Возвращает (значение, нужна ли ручная проверка), если строка полностью закрыта
    детерминированными/фразовыми правилами; иначе None (идём в LLM).
    """
    result = value.strip()

    # 0) атрибут исключён из реформатирования — возвращаем без изменений, LLM не вызываем.
    if _attribute_is_excluded(attribute, rule_set.excluded_attributes):
        return result, False

    # 1) точное фразовое совпадение (phrase_rules + бесчисловые deterministic).
    phrase_keys = {}
    for k, v in rule_set.phrase_map.items():
        phrase_keys[_normalize_phrase(k)] = v
    for k, v in rule_set.deterministic.items():
        if not re.search(r"\d", k):  # «чистые» фразы без числа
            phrase_keys[_normalize_phrase(k)] = v

    if phrase_keys:
        parts = result.split(";")
        mapped = []
        ok = True
        for part in parts:
            p = part.strip()
            if not p:
                mapped.append("")
                continue
            code = phrase_keys.get(_normalize_phrase(p))
            if code is None:
                ok = False
                break
            mapped.append(code)
        if ok:
            return "; ".join(mapped), False

    # 2) сопоставление с закрытыми регистрами, соответствующими атрибуту.
    prefixes = rule_set.attribute_to_register.get(attribute.strip().lower())
    if prefixes:
        table: dict[str, str] = {}
        for p in prefixes:
            table.update(rule_set.closed_registers.get(p, {}))
        if table:
            # значение через ';' — перечисление независимых позиций: каждая ищется отдельно,
            # иначе целиком оно ни одному пункту реестра не соответствует по определению
            mapped, needs_review = [], False
            for part in (p.strip() for p in result.split(";")):
                if not part:
                    continue
                matched = match_closed_register(part, table, prefix=prefixes[0],
                                                model=model, attribute=attribute)
                if matched is None:
                    break
                code, exact = matched
                mapped.append(code)
                needs_review = needs_review or not exact
            if mapped:
                return "; ".join(mapped), needs_review

    # 3) keyword → root: всё значение целиком должно быть «keyword число» / «число keyword».
    for keyword in sorted(rule_set.deterministic, key=len, reverse=True):
        # attribute-aware: применяем keyword только к его атрибуту-владельцу
        if not _attribute_matches(attribute, rule_set.deterministic_attributes.get(keyword, "")):
            continue
        root = rule_set.deterministic[keyword]
        token = rf"(?:{re.escape(keyword)}|{re.escape(root)})"
        num = r"(\d+[A-Za-z]*)"
        m = (re.fullmatch(rf"\s*{token}\s*{num}\s*", result, re.IGNORECASE)
             or re.fullmatch(rf"\s*{num}\s*{token}\s*", result, re.IGNORECASE))
        if m:
            num_part = m.group(1)
            nf = rule_set.deterministic_order.get(keyword, rule_set.number_first)
            return ((num_part + root) if nf else (root + num_part)), False

    return None


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


def start_row(state: GraphState, config: RunnableConfig) -> GraphState:

    update: dict = {}
    data_frame = state["data_frame"]
    rule_set = config["configurable"]["rule_set"]
    model = config["configurable"]["model"]

    # --- фиксируем результат ПРЕДЫДУЩЕЙ строки ---
    if state["index_row_table"] != 0:
        # needs_review — только пометка для ревьюера (жёлтая заливка), значение не подменяет
        needs_review = False
        if state.get("deterministic_result") is not None:
            last_answer = state["deterministic_result"]
            # либо исход требует регистрации, либо код присуждён по смыслу, а не по формулировке
            needs_review = _is_pending_result(last_answer) or state.get("needs_review_flag", False)
            rules_value = last_answer  # pending/ambiguous-сообщение остаётся в ячейке
        elif state.get("answer") is not None:
            last_answer = state["answer"]
            is_ambiguous = _is_ambiguous_answer(last_answer)
            # спорную по правилам строку возвращаем как записана, но помечаем;
            # расхождение сэмплов между собой — тоже повод показать строку человеку
            needs_review = is_ambiguous or state.get("needs_review_flag", False)
            rules_value = state["original_value"] if is_ambiguous else last_answer
            if not is_ambiguous:
                rules_value = _strip_redundant_parenthetical(rules_value, state["original_value"])
                # финальная строка собирается детерминированным форматтером, если для атрибута
                # из документа извлечён жёсткий шаблон рендера (знаки после запятой + суффикс)
                template = _match_render_template(state["attribute"], rule_set.render_templates,
                                                  state.get("rule_group_names"))
                if template:
                    rules_value = _apply_render_template(rules_value, template)
        else:
            rules_value = state["original_value"]
        new_row = {
            "Attribute": state["attribute"],
            "Value": state["original_value"],
            "Rules_Value": rules_value,
            "Ambiguous": needs_review,
            "Context": state.get("row_context", ""),
            "Deterministic": state.get("deterministic_result") is not None,
        }
        update["new_rows"] = [new_row]

    # --- настраиваем ТЕКУЩУЮ строку ---
    if state["index_row_table"] < data_frame.shape[0]:
        attribute = data_frame.iloc[state["index_row_table"]]["Attribute"]
        value = data_frame.iloc[state["index_row_table"]]["Value"]

        context = _row_context(data_frame, state["index_row_table"])
        update["attribute"] = attribute
        update["original_value"] = value
        update["row_context"] = context
        update["answer"] = None
        update["needs_review_flag"] = False
        update["messages"] = [RemoveMessage(id=k.id) for k in state["messages"]]

        det = apply_deterministic(value, rule_set, attribute, model=model)
        if det is not None:
            det_value, det_needs_review = det
            # полностью детерминировано — общий LLM-путь не нужен
            update["deterministic_result"] = det_value
            update["needs_review_flag"] = det_needs_review
            update["value"] = det_value
            update["rules_text"] = ""
            update["rule_group_names"] = []
            logger.info("=" * 80)
            logger.info(
                "Row %d/%d | Attribute: %r | Value: %r | DETERMINISTIC -> %r",
                state["index_row_table"] + 1, data_frame.shape[0], attribute, value, det_value,
            )
        else:
            update["deterministic_result"] = None
            update["value"] = value
            update["rules_text"], update["rule_group_names"] = build_rules_text(
                rule_set, attribute, value, context)
            logger.info("=" * 80)
            logger.info(
                "Row %d/%d | Attribute: %r | Value: %r | rules: %d chars",
                state["index_row_table"] + 1, data_frame.shape[0], attribute, value,
                len(update["rules_text"]),
            )

    return update


def _invoke_with_tools(model, messages):
    """Инвокает модель, прогоняя цикл tool-calls (для юнит-конверсий)."""
    response = model.invoke(messages)
    while getattr(response, "tool_calls", None):
        tool_messages = []
        for tc in response.tool_calls:
            tool_fn = TOOLS_BY_NAME.get(tc["name"])
            if tool_fn is None:
                content = "unknown tool"
            else:
                try:
                    content = str(tool_fn.invoke(tc["args"]))
                except Exception as e:
                    content = f"error: {e}"
            tool_messages.append(ToolMessage(content=content, tool_call_id=tc["id"]))
        messages = messages + [response] + tool_messages
        response = model.invoke(messages)
    return response


def _normalize_once(model, attribute: str, value: str, rules_text: str,
                    context: str = "") -> str:
    """Один сэмпл: применение правил + проверка кандидата на overreach.

    Правила подаются целиком за раз, а не порциями: при инкрементальной подаче ранний
    (возможно нерелевантный) чанк закрепляет ответ, и последующие правила уже только
    дополняют его, не пересматривая.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT1},
        {"role": "user", "content": (f"Attribute: {attribute}\n{_context_line(context)}"
                                     f"Value: {value}\n\n"
                                     f"Rules:\n{rules_text}")},
    ]
    candidate = _parse_result(_invoke_with_tools(model, messages).content).strip()
    if not candidate or _is_ambiguous_answer(candidate):
        return candidate

    # второй проход: precondition, исключения, утечка из worked-example, полнота
    messages = [
        {"role": "system", "content": VERIFY_PROMPT},
        {"role": "user", "content": (f"Attribute: {attribute}\n{_context_line(context)}"
                                     f"Original value: {value}\n"
                                     f"Candidate answer: {candidate}\n\n"
                                     f"Rules:\n{rules_text}")},
    ]
    checked = _parse_result(_invoke_with_tools(model, messages).content).strip()
    if checked and checked != candidate:
        logger.info("  [VERIFY] %r -> %r", candidate, checked)
    return checked or candidate


def normalize_row(state: GraphState, config: RunnableConfig) -> GraphState:
    """Ответ модели по полному набору правил, устойчивый к разбросу между прогонами.

    Спорные строки модель трактует по-разному от запуска к запуску, поэтому берётся
    несколько независимых сэмплов и большинство: это и повышает точность, и делает
    результат воспроизводимым (одна и та же таблица даёт одну и ту же выдачу).
    """
    model = config["configurable"]["model"]
    samples = max(1, int(config["configurable"].get("self_consistency", SELF_CONSISTENCY)))
    args = (model, state["attribute"], state["value"], state["rules_text"],
            state.get("row_context", ""))

    if samples == 1:
        candidates = [_normalize_once(*args)]
    else:
        with ThreadPoolExecutor(max_workers=samples) as pool:
            candidates = list(pool.map(lambda _: _normalize_once(*args), range(samples)))

    # большинство; при равенстве — вариант, встретившийся раньше (стабильный tie-break)
    counts = Counter(candidates)
    answer = max(candidates, key=lambda c: (counts[c], -candidates.index(c)))
    if len(counts) > 1:
        logger.info("  [VOTE] %r -> %r", dict(counts), answer)
    logger.info("  [NORMALIZE] %r -> %r", state["value"], answer)

    # разошедшиеся сэмплы = строка, которую правила документа не решают однозначно:
    # значение берём по большинству, но помечаем строку на ручную проверку
    return {"answer": answer, "needs_review_flag": len(counts) > 1}


def next_row(state: GraphState) -> GraphState:
    return {"index_row_table": state["index_row_table"] + 1}


def condition_after_start_row(state: GraphState) -> Literal["normalize_row", "next_row", "end_node"]:
    if state["index_row_table"] >= state["data_frame"].shape[0]:
        return "end_node"
    if state.get("deterministic_result") is not None:
        return "next_row"
    return "normalize_row"


def end_node(state: GraphState) -> GraphState:
    return {}


builder = StateGraph(GraphState)

builder.add_node("start_row", start_row)
builder.add_node("normalize_row", normalize_row)
builder.add_node("next_row", next_row)
builder.add_node("end_node", end_node)

builder.add_edge(START, "start_row")
builder.add_conditional_edges("start_row", condition_after_start_row)
builder.add_edge("normalize_row", "next_row")
builder.add_edge("next_row", "start_row")
builder.add_edge("end_node", END)

graph = builder.compile()


def run_single_rule_set(data_frame: pd.DataFrame, model, rule_set) -> pd.DataFrame:
    initial_state = {
        "messages": [],
        "index_row_table": 0,
        "data_frame": data_frame,
        "rules_text": "",
        "rule_group_names": [],
        "row_context": "",
        "new_rows": [],
        "original_value": "",
        "deterministic_result": None,
        "answer": None,
        "needs_review_flag": False,
    }
    final_state = graph.invoke(
        initial_state,
        config={
            "configurable": {"model": model, "rule_set": rule_set},
            # на строку приходится до 3 шагов графа (start_row/normalize_row/next_row)
            "recursion_limit": 3 * data_frame.shape[0] + 25,
        },
    )
    rows = _reconcile_equivalent_rows(final_state["new_rows"], rule_set, model)
    return pd.DataFrame(_canonicalize_formatting(rows))


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache
def get_app_config() -> AppConfig:
    return build_app_config(Settings())


def _write_excel_with_highlight(df: pd.DataFrame, ambiguous_cols: dict[str, list[bool]]) -> io.BytesIO:
    """Пишет DataFrame в Excel, заливая спорные ячейки жёлтым."""
    from openpyxl import Workbook
    from openpyxl.styles import PatternFill

    wb = Workbook()
    ws = wb.active
    yellow = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")

    ws.append(list(df.columns))
    for _, row in df.iterrows():
        ws.append(row.tolist())

    col_index = {name: i for i, name in enumerate(df.columns, start=1)}
    for col_name, flags in ambiguous_cols.items():
        c = col_index[col_name]
        for r, is_amb in enumerate(flags, start=2):
            if is_amb:
                ws.cell(row=r, column=c).fill = yellow

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf


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


@app.post("/process")
def process_excel(
        file: UploadFile = File(...),
        app_config: AppConfig = Depends(get_app_config),
):
    contents = file.file.read()
    data_frame = pd.read_excel(io.BytesIO(contents))

    # в выдаче сохраняем все исходные колонки: уточняющие колонки нужны ревьюеру
    result_df = data_frame.copy()
    ambiguous_cols: dict[str, list[bool]] = {}
    # документы независимы друг от друга — считаем их колонки параллельно
    with ThreadPoolExecutor(max_workers=max(1, len(app_config.rule_sets))) as pool:
        col_dfs = list(pool.map(
            lambda rs: run_single_rule_set(data_frame, app_config.model, rs),
            app_config.rule_sets,
        ))
    for rule_set, col_df in zip(app_config.rule_sets, col_dfs):
        result_df[rule_set.name] = col_df["Rules_Value"].to_list()
        if "Ambiguous" in col_df:
            ambiguous_cols[rule_set.name] = col_df["Ambiguous"].to_list()

    check_consistency(result_df, [rs.name for rs in app_config.rule_sets],
                      [_row_context(data_frame, i) for i in range(len(data_frame))])

    output_buffer = _write_excel_with_highlight(result_df, ambiguous_cols)

    return Response(
        content=output_buffer.read(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=result.xlsx"},
    )
