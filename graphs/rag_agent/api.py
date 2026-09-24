import io
import os
import re
import json
import base64
import logging
from rapidfuzz import fuzz
from uuid import uuid4
from functools import lru_cache
from typing import Literal, Annotated
from operator import add
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from dotenv import load_dotenv

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Depends, UploadFile, File, Form
from fastapi.responses import Response

from langgraph.graph import START, StateGraph, MessagesState, END
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import RemoveMessage, ToolMessage

from .modules.config import (AppConfig, Settings, build_app_config, hybrid_search,
                             RULE_SET_LIBRARY, document_id, value_shape)
from .modules.system_prompts import (SYSTEM_PROMPT1, VERIFY_PROMPT, REGISTER_PROMPT,
                                     RECONCILE_PROMPT, SUBJECT_CHOICE_PROMPT,
                                     SCOPE_PROMPT)
from .modules.agent import build_tools, run_agent
from .modules.chat import AdHocRule, format_adhoc_rules, history_messages
from .modules.workspace import get_workspace
from .modules.tools import array_of_tools, unit_mismatch, is_unit

TOOLS_BY_NAME = {t.name: t for t in array_of_tools}

load_dotenv()

logger = logging.getLogger("normalization_pipeline")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(handler)

RETRIEVAL_K = 4
# Сколько групп добавить по форме значения, когда имя атрибута не привело ни к одной.
MAX_SHAPE_GROUPS = 2
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
    subject_conflict: bool
    applied: str


SUBJECT_WORD_THRESHOLD = 85


def _words_match(left: str, right: str) -> bool:
    """Одно ли это слово с точностью до формы: «valve»/«valves», «bolt»/«bolting»."""
    if fuzz.ratio(left, right) >= SUBJECT_WORD_THRESHOLD:
        return True
    shorter, longer = sorted((left, right), key=len)
    return len(shorter) >= 4 and longer.startswith(shorter)


def _subject_matches(context: str, subject: str) -> bool:
    """Относится ли контекст строки к этому предмету.

    Разметка может перечислять несколько разновидностей через разделитель («pipe/tube») —
    достаточно совпадения с одной из них. Внутри разновидности сравнение пословное и нестрогое:
    в разметке «valves», в таблице «Gate Valve». Сравнивать строки целиком нельзя — «gate valve»
    и «ball valve» тогда неотличимы, поэтому каждое слово разновидности должно найти себе
    соответствие в контексте.
    """
    context_words = re.findall(r"\w+", str(context).lower())
    if not context_words:
        return False
    for alternative in re.split(r"[/,;|]", str(subject).lower()):
        subject_words = re.findall(r"\w+", alternative)
        if subject_words and all(any(_words_match(sw, cw) for cw in context_words)
                                 for sw in subject_words):
            return True
    return False


_SUBJECT_CHOICE_CACHE: dict[tuple[str, tuple[str, ...]], str] = {}


def _subject_by_model(model, context: str, subjects: tuple[str, ...]) -> str:
    """Спрашивает модель, к какому из предметов документа относится запись строки.

    Нужно там, где словами не сойтись: в таблице «90deg Elbow», а документ называет это
    «butt-welding fitting». Ответ кэшируется по паре (запись, набор предметов) — в таблице
    десятки строк на одно и то же изделие, и переспрашивать про каждую не нужно.
    """
    key = (context.strip().lower(), subjects)
    if key in _SUBJECT_CHOICE_CACHE:
        return _SUBJECT_CHOICE_CACHE[key]

    listing = "\n".join(f"- {s}" for s in subjects)
    messages = [
        {"role": "system", "content": SUBJECT_CHOICE_PROMPT},
        {"role": "user", "content": f"Recorded item: {context}\n\nEquipment kinds:\n{listing}"},
    ]
    try:
        raw = str(model.invoke(messages).content).strip().splitlines()[0].strip(" -*`\"'.")
    except Exception as e:
        logger.warning("  [SUBJECT] выбор предмета не удался: %s", e)
        raw = ""
    choice = next((s for s in subjects if s.lower() == raw.lower()), "")
    logger.info("  [SUBJECT] %r -> предмет %r", context, choice or "не определён")
    _SUBJECT_CHOICE_CACHE[key] = choice
    return choice


_SCOPE_CACHE: dict[tuple[str, str], bool] = {}
# Сколько текста глобальных правил показываем модели: область действия стоит в начале документа.
SCOPE_CHARS = 1500
# Сколько раз спросить модель об области действия; изделие чужое, только если все сказали «нет».
SCOPE_VOTES = 3


def _document_covers_row(rule_set, context: str, model=None) -> bool:
    """Относится ли этот документ к изделию строки вообще.

    Раньше смотрели только на теги групп (subjects.json): документ, который в своей области
    действия прямо называет регулирующие клапаны, отказывался их нормализовать, потому что ни
    одна группа не была помечена этим изделием, — а документ вовсе без разметки принимал любое
    изделие. Поэтому решает текст области действия, а ответ кэшируется: в таблице десятки
    строк на одно и то же изделие.
    """
    subjects = tuple(sorted({s for s in rule_set.group_subjects.values() if s}))
    if not context or model is None:
        return True  # строка без контекста — судить не о чем
    if any(_subject_matches(context, s) for s in subjects):
        return True

    key = (rule_set.name, context.strip().lower())
    if key in _SCOPE_CACHE:
        return _SCOPE_CACHE[key]
    if subjects and _subject_by_model(model, context, subjects):
        _SCOPE_CACHE[key] = True
        return True

    # Теги групп в вопрос не идут: помечены обычно лишь несколько групп, и список из одного
    # «rotary lobe blower» модель читала как «документ только про воздуходувки» — винтовой
    # компрессор из области действия получал пустые ячейки. Решает текст области действия.
    messages = [
        {"role": "system", "content": SCOPE_PROMPT},
        {"role": "user", "content": (f"Recorded item: {context}\n\nOpening rules of the standard:\n"
                                     f"{rule_set.global_rules[:SCOPE_CHARS]}")},
    ]

    def ask(_):
        try:
            return str(model.invoke(messages).content).strip().upper()
        except Exception as e:
            logger.warning("  [SCOPE] вопрос об области действия не удался: %s", e)
            return "YES"

    # пустая колонка — самое разрушительное решение пайплайна, а ответ модели на варианте
    # названия («Screw Chiller» против «water chillers») колебался 50/50: изделие считается
    # чужим, только если так ответили все опросы
    with ThreadPoolExecutor(max_workers=SCOPE_VOTES) as pool:
        answers = list(pool.map(ask, range(SCOPE_VOTES)))
    covered = not all(a.startswith("NO") for a in answers)
    logger.info("  [SCOPE] %r в области действия %r: %s", context, rule_set.name, covered)
    _SCOPE_CACHE[key] = covered
    return covered


def _equipment_kind(rule_set, context: str, model=None) -> str:
    """Вид изделия этой строки, когда контекст называет его не словами, а кодом.

    Документ на несколько изделий разводит правила (и строки своих таблиц) по видам, а в
    датащите стоит «Item: GLV-15001». Пайплайн этот код разгадывает — но раньше только для
    отбора групп, а модель дальше выбирала строку таблицы наугад: «7,62 cm» уезжало в «0.1 m»
    по правилу для труб. Ответ кэшируется в _subject_by_model, так что вопрос один на контекст.
    """
    subjects = tuple(sorted({s for s in rule_set.group_subjects.values() if s}))
    if len(subjects) < 2 or not context or model is None:
        return ""
    if any(_subject_matches(context, s) for s in subjects):
        return ""  # контекст называет вид словами — модель прочтёт его сама
    return _subject_by_model(model, context, subjects)


def _select_by_subject(rule_set, pairs: list[tuple[str, str]], context: str,
                       model=None) -> tuple[list[tuple[str, str]], bool]:
    """Отсеивает найденные группы по предмету строки; сообщает о неразрешимом смешении.

    Если документ описывает несколько изделий, одинаково названные атрибуты означают разное.
    Группы без предмета относятся ко всему документу и остаются всегда. Из остальных остаются
    те, чей предмет совпал с записью строки; если не совпал ни один — правила остаются все, но
    строка идёт к ревьюеру, а не решается подбрасыванием монетки.
    """
    subject_of = {name: rule_set.group_subjects.get(name, "") for name, _ in pairs}
    concrete = {s for s in subject_of.values() if s}
    if len(concrete) < 2:
        return pairs, False

    matched = [p for p in pairs if _subject_matches(context, subject_of[p[0]])]
    if not matched and context:
        # словами не сошлось — разовый вопрос модели, к какому предмету относится изделие
        choice = _equipment_kind(rule_set, context, model)
        if choice:
            matched = [p for p in pairs if _subject_matches(choice, subject_of[p[0]])]

    if matched:
        kept = [p for p in pairs if not subject_of[p[0]] or p in matched]
        logger.info("  [SUBJECT] %r -> оставлены правила предметов %s",
                    context, sorted({subject_of[n] for n, _ in kept if subject_of[n]}))
        return kept, False

    logger.warning("  [SUBJECT] найдены правила разных предметов %s, контекст %r не уточняет",
                   sorted(concrete), context)
    return pairs, True


def _adhoc_rules_for_attribute(rules: list[AdHocRule], attribute: str) -> list[AdHocRule]:
    """Правила пользователя, относящиеся к этой строке: общие + названные для её атрибута.

    Имя атрибута пользователь пишет своими словами («материал» против «Material designation»),
    поэтому сравнение то же пословное, что и для владельцев правил документа.
    """
    return [r for r in rules if not r.attribute or _attribute_matches(attribute, r.attribute)]


def build_rules_text(rule_set, attribute: str, value: str, context: str = "",
                     top_k: int = RETRIEVAL_K, model=None) -> tuple[str, list[str], bool]:
    """Полный набор правил для одной строки: global + релевантные local-группы.

    Ретрив видит атрибут, само значение (в нём может быть признак исключения — префикс, скобка,
    диапазон) и контекст строки. Возвращает (текст правил, имена групп, смешаны ли предметы).
    """
    query = " ".join(x for x in (attribute, str(value), context) if x).strip()
    body_to_name = {g.body: g.attribute for g in rule_set.groups}
    pairs = [(body_to_name.get(body, ""), body)
             for body in hybrid_search(rule_set, query, top_k=top_k)]
    pairs, subject_conflict = _select_by_subject(rule_set, pairs, context, model)

    # Группа с именем (или алиасом) самого атрибута — это и есть правило для него; ранг
    # похожести тут ничего не решает, а без неё строка остаётся без своего правила.
    named = False
    for group in rule_set.groups:
        names = [group.attribute] + list(group.aliases)
        if any(n and _attribute_matches(attribute, n) for n in names):
            named = True
            pair = (group.attribute, group.body)
            if pair not in pairs:
                logger.info("  [SEARCH] группа по имени атрибута: %r", group.attribute)
            else:
                # уже найдена по похожести, но не первой: вперёд её, иначе шаблон рендера
                # соседней группы с той же единицей («cooling capacity», 0 зн. kW) побеждал
                # собственный шаблон атрибута («Fan power» → «electrical power», 1 зн.)
                pairs.remove(pair)
            pairs.insert(0, pair)
            # правило именно этого атрибута найдено: спор о том, к какому изделию относятся
            # соседние группы, строку больше не решает
            subject_conflict = False

    # Имя атрибута ни к одной группе не привело («Precharge», «Plate size»): тогда правило
    # узнаётся по форме значения — пример документа «"2.5 bar" → "250 kPa"» той же формы, что
    # и запись «1.5 bar». Только как запасной путь: у названной группы свой пример главнее.
    shape = value_shape(value)
    if not named and shape:
        by_shape = [g for g in rule_set.groups if shape in g.shapes][:MAX_SHAPE_GROUPS]
        for group in by_shape:
            pair = (group.attribute, group.body)
            if pair not in pairs:
                logger.info("  [SEARCH] группа по форме значения %r: %r", shape, group.attribute)
                pairs.insert(0, pair)

    blocks: list[str] = []
    kind = _equipment_kind(rule_set, context, model)
    if kind:
        # то, что пайплайн уже выяснил про строку, должно дойти и до модели: таблицы документа
        # разведены по видам изделий, а в контексте стоит код изделия, а не его название
        blocks.append(f"The item recorded in this row is a {kind}. Where a rule or a table of "
                      f"this document distinguishes kinds of equipment, this is the kind whose "
                      f"entry applies.")
    if rule_set.global_rules:
        blocks.append(rule_set.global_rules)
    names: list[str] = []
    for idx, (name, body) in enumerate(pairs):
        logger.info("  [SEARCH] Rule %d: %r | %r", idx + 1, name, body[:100])
        if name:
            names.append(name)
        if body not in blocks:
            blocks.append(body)

    if not blocks:
        return ("No relevant rules were found in the knowledge base for this attribute.",
                names, subject_conflict)

    text = "\n".join(blocks)
    expanded = "\n\n".join([text] + expand_references(rule_set, text, attribute, names, str(value)))
    return expanded, names, subject_conflict


_CLAUSE_DEF_RE = re.compile(r"^\s*(\d+(?:\.\d+)+)\s")
_CLAUSE_REF_RE = re.compile(r"\b\d+(?:\.\d+)+\b")
# Упоминание таблицы/приложения/реестра — признак, что правило отсылает к списку значений.
_TABLE_REF_RE = re.compile(r"\b(?:table|appendix|register|annex)\b", re.I)
MAX_PULLED_CLAUSES = 12


def _context_line(context: str) -> str:
    """Строка контекста для промпта; пустой контекст не добавляет ничего."""
    return f"Row context: {context}\n" if context else ""


def _register_lines_for_attribute(rule_set, attribute: str, group_names: list[str] | None = None,
                                  value: str = "") -> tuple[str, str]:
    """Строки таблиц документа для этого атрибута: (keyword→root, фраза→код).

    Правило говорит «взять корень из Table 1», но самой таблицы в тексте правил нет — она
    лежит отдельным артефактом извлечения. Без неё модели нечего подставить, и она берёт
    первое похожее значение из примеров соседнего атрибута.

    Возвращаются два разных вида таблиц, и смешивать их нельзя: keyword→root — корень для
    ОБОЗНАЧЕНИЯ («Nominal pipe size» → «NPS»), а фраза→код — закрытый список значений.
    Поданные вместе под заголовком «выбери запись из списка», корни становятся ответом:
    размер «7,62 cm» так превратился в «NPS 7,62 cm».
    """
    # Подпись атрибута в таблице заказчика и имя группы правил — разные тексты («Facing type»
    # против «end connection»), поэтому владельца ищем и среди групп, которые нашёл ретрив.
    candidates = [attribute] + list(group_names or [])
    roots: list[str] = []
    lines: list[str] = []
    for keyword, root in rule_set.deterministic.items():
        owner = rule_set.deterministic_attributes.get(keyword, "")
        if owner and any(_attribute_matches(c, owner) for c in candidates):
            roots.append(f"{keyword} -> {root}")
    prefixes: list[str] = []
    for candidate in candidates:
        prefixes.extend(_registers_for_attribute(rule_set, candidate))
    if not prefixes and value:
        prefixes = _registers_by_value(rule_set, value)
    for prefix in dict.fromkeys(prefixes):
        for phrase, code in rule_set.closed_registers.get(prefix, {}).items():
            lines.append(f"{phrase} -> {code}")
    return "\n".join(dict.fromkeys(roots)), "\n".join(dict.fromkeys(lines))


REGISTER_CONTENT_THRESHOLD = 70


def _registers_by_value(rule_set, value: str) -> list[str]:
    """Ищет реестр по самому значению, когда по имени атрибута он не нашёлся.

    Имена в документе и в таблице заказчика могут не пересекаться вовсе («Facing type» против
    группы «end connection»), а вот запись «Face, raised» на пункт реестра похожа. Найдя пункт,
    показываем весь реестр его атрибута-владельца: выбирать нужно из полного списка.
    """
    normalized = _normalize_phrase(value)
    best_prefix, best_score = "", 0
    for prefix, table in rule_set.closed_registers.items():
        for phrase in table:
            score = fuzz.token_sort_ratio(normalized, _normalize_phrase(phrase))
            if score > best_score:
                best_prefix, best_score = prefix, score
    if best_score < REGISTER_CONTENT_THRESHOLD:
        return []
    owners = [prefixes for prefixes in rule_set.attribute_to_register.values()
              if best_prefix in prefixes]
    logger.info("  [REFS] реестр найден по значению %r (совпадение %d%%)", value, best_score)
    return owners[0] if owners else [best_prefix]


# Ссылка на пункт ДРУГОГО документа: «per 6.2 of OGX-STD 410», «Clauses 2.1 and 2.2 of XYZ STD 100»,
# «Clause 6 of O GX-STD 410» (распознавание разрывает обозначение пробелом — сравниваем без них).
_NUMS = r"\d+(?:\.\d+)*(?:\s*(?:,|and|or)\s*\d+(?:\.\d+)*)*"
_EXTERNAL_REF_RE = re.compile(
    r"(?i:clauses?|sections?|per|under|in)\s+(?P<nums>" + _NUMS + r")\s*,?\s+(?i:of)\s+"
    r"(?P<doc>[A-Z][A-Za-z0-9 ._/-]{1,24}?\d{2,5})\b")
# и обратный порядок: «under XYZ STD 100 2.3», «per OGX-STD 410, 6.2»
_EXTERNAL_REF_REV_RE = re.compile(
    r"(?P<doc>\b[A-Z][A-Z0-9 _/-]{1,20}?\d{2,5})\s*,?\s*(?i:clauses?\s+|sections?\s+)?"
    r"(?P<nums>\d+\.\d+(?:\.\d+)*)\b")
MAX_EXTERNAL_CLAUSES = 8


def _external_references(rule_set, text: str) -> list[tuple[str, list[str], tuple[int, int]]]:
    """Ссылки текста правил на пункты других документов: (обозначение, номера, позиция).

    Ссылка на самого себя («per 7.1 of HVX-STD 134» внутри HVX-STD 134) — обычная внутренняя.
    Упоминание документа без номера пункта («in the same manner as OGX-STD 410») ссылкой на
    правило не считается: подтянуть по нему нечего.
    """
    own = document_id(rule_set.name)
    found: list[tuple[str, list[str], tuple[int, int]]] = []
    for pattern in (_EXTERNAL_REF_RE, _EXTERNAL_REF_REV_RE):
        for m in pattern.finditer(str(text)):
            doc = m.group("doc").strip()
            key = document_id(doc)
            if (key == own or not re.search(r"[A-Z]{2}", doc.replace(" ", ""))
                    or re.match(r"(?:table|appendix|annex|clause|section|figure)\b", doc, re.I)):
                continue
            if pattern is _EXTERNAL_REF_REV_RE and key not in RULE_SET_LIBRARY:
                continue  # обратный порядок слишком похож на обычный текст — только известные документы
            if any(start <= m.start() < end for _, _, (start, end) in found):
                continue
            nums = re.findall(r"\d+(?:\.\d+)*", m.group("nums"))
            found.append((doc, nums, m.span()))
    return found


def _without_external_references(rule_set, text: str) -> str:
    """Текст правил, где ссылки на чужие пункты заменены пробелами.

    Иначе «per 6.2 of OGX-STD 410» разрешалось как СВОЙ пункт 6.2 — в строку про температуру
    приезжало правило давления этого же документа.
    """
    chars = list(str(text))
    for _, _, (start, end) in _external_references(rule_set, text):
        chars[start:end] = " " * (end - start)
    return "".join(chars)


def _external_clause_lines(target, number: str) -> list[str]:
    """Текст пункта другого документа; для «Clause 6» — все его подпункты «6.x»."""
    index = target.clause_index
    lines = [index[number]] if number in index else []
    lines += [line for key, line in index.items() if key.startswith(number + ".")]
    return lines[:MAX_EXTERNAL_CLAUSES]


def _resolve_external(rule_set, text: str) -> tuple[list[str], list[str]]:
    """(цитаты чужих пунктов для промпта, ссылки, которые разрешить нечем)."""
    quoted: list[str] = []
    missing: list[str] = []
    for doc, nums, _ in _external_references(rule_set, text):
        target = RULE_SET_LIBRARY.get(document_id(doc))
        for number in nums:
            lines = _external_clause_lines(target, number) if target else []
            if lines:
                quoted.append(f"[{doc}, {number}]\n" + "\n".join(lines))
            else:
                missing.append(f"{number} документа {doc}"
                               + ("" if target else " (документа нет в базе)"))
    return list(dict.fromkeys(quoted)), list(dict.fromkeys(missing))


def expand_references(rule_set, text: str, attribute: str, group_names: list[str] | None = None,
                      value: str = "") -> list[str]:
    """Дотягивает то, на что найденные правила ссылаются: пункты по номеру и таблицы атрибута.

    Ссылка разрешается только если её номер реально определён в документе, поэтому десятичные
    числа в значениях («0.25 in») за ссылку не принимаются. Глубина — два уровня, с потолком
    по количеству, чтобы контекст строки не разрастался. Пункт другого загруженного документа
    («per 6.2 of OGX-STD 410») цитируется из того документа, а не из своего.
    """
    blocks: list[str] = []

    external, _ = _resolve_external(rule_set, text)
    if external:
        logger.info("  [REFS] подтянуто пунктов других документов: %d", len(external))
        blocks.append(
            "Clauses of OTHER documents that the rules above cite, quoted from those documents. "
            "Apply such a clause only as far as the citing rule adopts it (for the quantity it "
            "names); where the citing rule contrasts itself with it, the citing rule governs.\n"
            + "\n\n".join(external))

    text = _without_external_references(rule_set, text)
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
        roots, register = _register_lines_for_attribute(rule_set, attribute, group_names, value)
        if register:
            logger.info("  [REFS] подтянут закрытый реестр атрибута: %d строк",
                        len(register.splitlines()))
            blocks.append(
                "Register entries this document defines for this attribute. This list is "
                "complete and closed: use an entry from it, and never invent one that is "
                "not listed.\n" + register)
        if roots:
            logger.info("  [REFS] подтянуты корни обозначений: %d строк", len(roots.splitlines()))
            blocks.append(
                "Root words this document assigns to the NAMES of designations. A root "
                "replaces the name of a designation, never the value itself: use it only "
                "where the value is that designation together with its number, and never as "
                "a replacement for a measured quantity carrying a unit. This list is not a "
                "list of allowed values.\n" + roots)
    return blocks


_CITATION_RE = re.compile(r"(?:per|see|under|clause)\s+(\d+(?:\.\d+)+)", re.I)


def _own_rules_text(rule_set, attribute: str, group_names: list[str] | None,
                    rules_text: str) -> str:
    """Правила, которые относятся к атрибуту строки: группы, названные его именем или алиасом.

    Неразрешимая ссылка важна строке, только если её делает правило этого атрибута. Соседняя
    группа, попавшая в выдачу по похожести («Sound pressure level … per 4.2 of ZZX-STD 000»),
    не повод метить «Motor power». Если своей группы нет, судим по всему тексту, как раньше.
    """
    names = set(group_names or [])
    own = [g.body for g in rule_set.groups if g.attribute in names
           and any(n and _attribute_matches(attribute, n) for n in [g.attribute] + list(g.aliases))]
    return "\n".join(own) if own else rules_text


def _unresolved_references(rule_set, text: str) -> list[str]:
    """Пункты, на которые правила ссылаются, но чей текст в документ не извлёкся.

    Извлечение иногда теряет номер пункта, и тогда «normalized per 3.1» ведёт в никуда:
    модель видит отсылку и не видит правила. Сказать об этом честнее, чем молча не применить.
    """
    _, external_missing = _resolve_external(rule_set, str(text))
    text = _without_external_references(rule_set, str(text))
    present = {m.group(1) for line in text.splitlines() if (m := _CLAUSE_DEF_RE.match(line))}
    return [ref for ref in dict.fromkeys(_CITATION_RE.findall(text))
            if ref not in present and ref not in rule_set.clause_index] + external_missing


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


# Номер пункта стоит в начале объяснения либо после слова-указателя; «0.4 kV» в середине
# фразы — это значение, а не ссылка.
_CITED_CLAUSE_RE = re.compile(
    r"(?:^|(?:clause|per|see|under|and|,)\s+)(\d{1,2}(?:\.\d+)+)", re.I)


def _cited_clauses(payload: dict) -> str:
    """Номера пунктов, на которые сослалась модель: всё остальное она пересказывает."""
    entries = payload.get("applied")
    entries = entries if isinstance(entries, list) else [entries or ""]
    numbers: list[str] = []
    for entry in entries:
        numbers.extend(_CITED_CLAUSE_RE.findall(str(entry).strip()))
    numbers = list(dict.fromkeys(numbers))
    return "по пункту " + ", ".join(numbers[:4]) if numbers else ""


def _describe_change(value: str, answer: str) -> str:
    """Чем результат отличается от записи — по самой строке, без слов модели."""
    before, after = str(value).strip(), str(answer).strip()
    if before == after:
        return ""
    if before.replace(" ", "") == after.replace(" ", ""):
        return "убраны пробелы" if len(after) < len(before) else "добавлен пробел"
    if before.lower() == after.lower():
        return "изменён регистр"
    if before.replace(" ", "").lower() == after.replace(" ", "").lower():
        spaces = "добавлен пробел" if after.count(" ") > before.count(" ") else "убраны пробелы"
        return f"изменён регистр, {spaces}"
    if _alnum(before) == _alnum(after):
        was = "".join(dict.fromkeys(c for c in before if not c.isalnum() and not c.isspace()))
        now = "".join(dict.fromkeys(c for c in after if not c.isalnum() and not c.isspace()))
        if was != now:
            return f"разделители: «{was or 'нет'}» → «{now or 'нет'}»"
        return "изменено написание"
    was_quantity, now_quantity = _NUM_UNIT_RE.fullmatch(before), _NUM_UNIT_RE.fullmatch(after)
    if (was_quantity and now_quantity
            and unit_mismatch(was_quantity.group(3), "count")
            and unit_mismatch(now_quantity.group(3), "count")):
        if _unit_key(was_quantity.group(3)) != _unit_key(now_quantity.group(3)):
            return f"пересчёт: {before} → {after}"
        return f"округление: {before} → {after}"

    before_words = sorted(re.findall(r"\w+", before.lower()))
    after_words = sorted(re.findall(r"\w+", after.lower()))
    if before_words == after_words:
        return "переставлены части значения"
    dropped = [w for w in dict.fromkeys(re.findall(r"\w+", before)) if w.lower() not in after_words]
    added = [w for w in dict.fromkeys(re.findall(r"\w+", after)) if w.lower() not in before_words]
    parts = []
    if dropped:
        parts.append("убрано: " + ", ".join(f"«{w}»" for w in dropped[:3]))
    if added:
        parts.append("добавлено: " + ", ".join(f"«{w}»" for w in added[:3]))
    return "; ".join(parts) or f"значение заменено на «{after}»"


def _applied_text(payload: dict) -> str:
    """Перечень правил из ответа модели: список «applied» либо старое одиночное поле."""
    entries = payload.get("applied")
    if isinstance(entries, list):
        return "; ".join(str(e).strip() for e in entries if str(e).strip())
    return str(entries or payload.get("precondition", "")).strip()


def _parse_payload(answer) -> dict:
    """Ответ модели как объект: кроме result там лежит и объяснение, почему так."""
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", str(answer).strip()).strip()
    try:
        obj = json.loads(text)
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


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


def _match_render_template(attribute: str, render_templates: dict,
                           group_names: list[str] | None = None,
                           value: str = "") -> dict | None:
    """Находит шаблон рендера: сперва по группам, которые ретрив счёл релевантными для строки,
    затем по имени самого атрибута (имя группы + алиасы, подстрока).

    Имя атрибута в датащите может не иметь ничего общего с именем группы правил
    («Wear travel, in» ↔ группа «dimension»), поэтому ретрив здесь — основной путь. Но группа,
    спорящая с подписью атрибута по словам, — это другой атрибут, и её шаблон не наш:
    «nominal length» (м, 1 знак) не должен переписывать «Nominal size».
    """
    candidates = []
    for name in group_names or []:
        tmpl = render_templates.get(name) or render_templates.get(_norm_attr(name))
        # «Fan power» спорит словами с «electrical power», но это его алиас: не другой атрибут
        aliased = tmpl and any(_attribute_matches(attribute, a) for a in tmpl.get("aliases", []) if a)
        if tmpl and (aliased or not _names_clash(attribute, name)):
            candidates.append(tmpl)
    a = _norm_attr(attribute)
    if a:
        for attr, tmpl in render_templates.items():
            for c in [attr] + list(tmpl.get("aliases", [])):
                c_norm = _norm_attr(c)
                if c_norm and (c_norm in a or a in c_norm):
                    candidates.append(tmpl)
    if not candidates:
        return None
    # ретрив возвращает несколько групп подряд, и первая может быть про другую величину:
    # выбираем шаблон, чья единица совпадает с единицей самого значения
    m = _NUM_UNIT_RE.fullmatch(str(value).strip())
    if m:
        for tmpl in candidates:
            if _unit_key(tmpl.get("suffix", "")) == _unit_key(m.group(3)):
                return tmpl
    return candidates[0]


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


def _example_keeps_space(rules_text: str, root: str) -> bool:
    """Показывает ли пример документа этот корень с пробелом перед числом.

    «Class 150» в примере значит, что склеивать нельзя, как бы ни выглядели корни в других
    документах. Если примера с этим корнем нет, решает общая конвенция keyword→root.
    """
    for _, target in _EXAMPLE_RE.findall(str(rules_text)):
        if re.search(rf"\b{re.escape(root)}\s+\d", target, re.IGNORECASE):
            return True
    return False


def _align_root_spacing(answer: str, rule_set, rules_text: str = "") -> str:
    """Корень и число склеиваются так же, как их собирает детерминированный путь: без пробела.

    Один и тот же атрибут давал «TRIM8» в одной строке и «TRIM 12» в другой, «CL600» и
    «CL 600»: конвенцию документа знает keyword→root, а LLM-путь о ней не знает. Правится
    только пробел между корнем и числом, содержимое не трогается. Если же документ показал
    в примере пробел («150 Class» → «Class 150»), его пример и решает.
    """
    roots = {root for root in rule_set.deterministic.values() if root and root.isalpha()}
    if not roots:
        return answer
    token = "|".join(re.escape(r) for r in sorted(roots, key=len, reverse=True))
    number = r"\d+[A-Za-z]*"
    tail = r"(\s*(?:\([^()]*\)|\S.*))?"
    m = (re.fullmatch(rf"\s*({token})\s+({number}){tail}\s*", answer)
         or re.fullmatch(rf"\s*({number})\s+({token}){tail}\s*", answer))
    if not m:
        return answer
    if _example_keeps_space(rules_text, m.group(1)) or _example_keeps_space(rules_text, m.group(2)):
        return answer
    aligned = f"{m.group(1)}{m.group(2)}{m.group(3) or ''}"
    if aligned != answer:
        logger.info("  [ROOT] %r -> %r (конвенция документа: корень и число слитно)",
                    answer, aligned)
    return aligned


# Обозначение вида «PN 100»: буквенный префикс из 2–5 заглавных и число (в любом порядке).
_DESIGNATION_RE = re.compile(r"\s*([A-Z]{2,5})\s*-?\s*(\d+[A-Za-z]*)\s*")
_DESIGNATION_REV_RE = re.compile(r"\s*(\d+[A-Za-z]*)\s*-?\s*([A-Z]{2,5})\s*")


def _designation_parts(text: str) -> tuple[str, str] | None:
    """(префикс, число) для записи-обозначения; иначе None."""
    m = _DESIGNATION_RE.fullmatch(str(text))
    if m:
        return m.group(1).upper(), m.group(2)
    m = _DESIGNATION_REV_RE.fullmatch(str(text))
    return (m.group(2).upper(), m.group(1)) if m else None


def _letters_attached_to_digits(value: str) -> bool:
    """Все буквы записи приклеены к цифрам («10K», «15K»), отдельного слова-префикса нет.

    «PN 100» несёт собственное обозначение отдельным словом, а «10K» — это форма записи
    номера, и документ вправе задать ей корень.
    """
    return not re.search(r"(?<![\d])[A-Za-z]{2,}(?![\d])", str(value))


def _root_invented(value: str, answer: str, rule_set, attribute: str = "") -> bool:
    """Корень обозначения взят из документа, хотя запись это обозначение не называет.

    Корень заменяет имя обозначения: «Trim 8» → «TRIM8», «600#» → «CL600» (у записи вовсе нет
    букв, только номер и знак класса). Но «F51» — марка стали, и «TRIM51» из неё получается
    только выбрасыванием буквы; «PN 100» — класс по другой системе, и «CL100» его подменяет.
    Правило одно: буквы записи должны называть это же обозначение — сам корень или значащее
    слово его ключа.
    """
    parts = _designation_parts(answer)
    if not parts:
        return False
    root = parts[0]
    known_roots = {r.upper() for r in rule_set.deterministic.values() if r and r.isalpha()}
    if root not in known_roots:
        return False  # это не корень документа, а, например, код закрытого реестра
    markers = {root.lower()}
    for keyword, keyword_root in rule_set.deterministic.items():
        if str(keyword_root).upper() == root:
            markers.update(w.lower() for w in re.findall(r"[A-Za-z]{4,}", str(keyword)))
    letters = [w for w in re.findall(r"[A-Za-z]+", str(value))]
    if not letters:
        return False  # в записи одни цифры: имени обозначения там и не было
    names_it = any(w.lower() in markers or w.lower() in {m[:len(w)] for m in markers}
                   for w in letters)
    if names_it:
        return False
    # подпись атрибута называет это обозначение, а буквы записи — часть её номера («10K»):
    # документ вправе определить такую форму, и подменой обозначения это не является
    if attribute and _letters_attached_to_digits(value):
        attribute_words = {w.lower() for w in re.findall(r"[A-Za-z]{3,}", str(attribute))}
        if attribute_words & markers:
            return False
    return True


def _root_misapplied(answer: str, rule_set, value: str = "") -> bool:
    """Корень документа приклеен не к номеру обозначения, а к чему-то другому.

    Корень заменяет ИМЯ обозначения, за которым идёт его номер («Trim 8» → «TRIM8»). За ним не
    может стоять слово («TRIM f6a», «TRIM F51») и не может стоять измеренная величина: «150 MM»
    превращалось в «DN5.906 in», где к пересчитанной длине приписан корень условного прохода.
    """
    text = str(answer).strip()
    recorded_has_unit = bool(_NUM_UNIT_RE.fullmatch(str(value).strip()))
    for root in {r for r in rule_set.deterministic.values() if r and r.isalpha() and len(r) > 1}:
        # после корня не буква: «DN5.906» — корень с числом, а «BWR» — другое слово
        m = re.match(rf"^{re.escape(root)}(?![A-Za-z])[\s-]*(.*)$", text, re.IGNORECASE)
        if not m or not m.group(1):
            continue
        rest = m.group(1)
        if not rest[0].isdigit():
            return True
        if recorded_has_unit and _NUM_UNIT_RE.fullmatch(rest):
            return True
        if re.match(r"^\d+[.,]\d", rest):
            return True  # номер обозначения целый: «NPS6.000» — это пересчитанная длина
    return False


def _unit_invented(attribute: str, value: str, answer: str) -> bool:
    """Единица появилась из ниоткуда: её нет ни в записи, ни в подписи атрибута.

    «150» в графе «Nominal size» превращалось в «150.000 in.» — это уже утверждение о величине,
    а не о её написании. Документ вправе требовать единицу, но взять её неоткуда: решает человек.
    """
    a = _NUM_UNIT_RE.fullmatch(str(answer).strip())
    if not a or not unit_mismatch(a.group(3), "count"):
        return False  # ответ без единицы либо единица нечитаемая — судить не о чем
    if _NUM_UNIT_RE.fullmatch(str(value).strip()) or _LABEL_UNIT_RE.search(str(attribute)):
        return False
    return bool(re.fullmatch(r"\s*-?\d+(?:[.,]\d+)?\s*", str(value)))


# Единица не может начинаться с точки или запятой, иначе «9.5» разбирается как 9 + «.5».
_NUM_UNIT_RE = re.compile(r"\s*(-?\d+(?:[.,]\d+)?)(\s*)([^\d\s.,].*?)\s*")
_LABEL_UNIT_RE = re.compile(r"[,(]\s*([A-Za-z°]{1,6}\.?)\s*\)?\s*$")


def _unit_relabelled(attribute: str, value: str, answer: str) -> bool:
    """Единица подменена без пересчёта: число то же, а единица уже другая.

    «Wall thickness, mm» со значением «9.5» превращалось в «9.50 in.» — запись просто
    переобозначили дюймами. Если пересчёт был, меняется число; если не был, единица обязана
    остаться той, что названа в записи или в подписи атрибута.
    """
    a = _NUM_UNIT_RE.fullmatch(str(answer))
    if not a:
        return False
    v = _NUM_UNIT_RE.fullmatch(str(value))
    recorded_unit = v.group(3) if v else ""
    if not recorded_unit:
        label = _LABEL_UNIT_RE.search(str(attribute))
        recorded_unit = label.group(1) if label else ""
    if not recorded_unit or not unit_mismatch(recorded_unit, a.group(3)):
        return False
    if _unit_key(recorded_unit) == _unit_key(a.group(3)):
        return False  # отличие только в регистре или знаках — это приведение к форме, не подмена
    number = v.group(1) if v else str(value)
    try:  # «9.5» и «9.50» — одно и то же число, а вот единица уже другая
        return abs(float(str(number).replace(",", ".")) - float(a.group(1).replace(",", "."))) < 1e-9
    except ValueError:
        return _digit_groups(number) == _digit_groups(a.group(1))


_SPACING_RULE_RE = re.compile(
    r"(?:single|one|a)\s+space|separated\s+from\s+the\s+numera|followed\s+by\s+a\s+space", re.I)


def _document_mandates_spacing(rules_text: str) -> bool:
    """Требует ли документ пробел между числом и единицей явным правилом."""
    return bool(_SPACING_RULE_RE.search(str(rules_text)))


def _restore_unit_spacing(value: str, answer: str) -> str:
    """Пробел между числом и единицей берётся из записи, если единица та же.

    Документ задаёт единицу и знаки после запятой, но молчит про пробел; модель пишет
    «16.00mm2» там, где записано «16 mm2». Где правила молчат, формат записи и сохраняем.
    """
    v = _NUM_UNIT_RE.fullmatch(str(value))
    a = _NUM_UNIT_RE.fullmatch(str(answer))
    if not v or not a or _unit_key(v.group(3)) != _unit_key(a.group(3)):
        return answer
    if not unit_mismatch(a.group(3), "count"):
        return answer  # «- GP - 2» из номера модели — не единица, пробелы там не наши
    if bool(v.group(2)) == bool(a.group(2)):
        return answer
    return f"{a.group(1)}{' ' if v.group(2) else ''}{a.group(3)}"


_WORD_RE = re.compile(r"\w+")
# Пример правила: «"0.075%" -> "0.075 % of span"». Слова, появившиеся справа, документ велит
# дописать; слова, которые были и слева, он лишь переставляет.
_EXAMPLE_RE = re.compile(r"[\"“']([^\"“”']{1,120})[\"”']\s*(?:->|→|becomes)\s*[\"“']([^\"“”']{1,120})[\"”']")


def _mandated_additions(rules_text: str) -> set:
    """Слова, которые примеры документа добавляют к записи: их удалять нельзя."""
    added = set()
    for source, target in _EXAMPLE_RE.findall(str(rules_text)):
        source_words = {w.lower() for w in _WORD_RE.findall(source)}
        added.update(w.lower() for w in _WORD_RE.findall(target)
                     if w.isalpha() and w.lower() not in source_words)
    return added


def _document_word_tokens(rule_set, attribute: str, group_names: list[str] | None = None) -> set:
    """Слова, которые документ вправе внести в ответ: корни и коды его таблиц."""
    tokens = set()
    for root in rule_set.deterministic.values():
        tokens.update(w.lower() for w in _WORD_RE.findall(str(root)))
    for table in rule_set.closed_registers.values():
        for code in table.values():
            tokens.update(w.lower() for w in _WORD_RE.findall(str(code)))
    template = _match_render_template(attribute, rule_set.render_templates, group_names)
    if template:
        tokens.update(w.lower() for w in _WORD_RE.findall(str(template.get("suffix", ""))))
    return tokens


def _strip_added_words(value: str, answer: str, allowed: set) -> str:
    """Убирает слова, дописанные к записи: их нет ни в ней, ни в таблицах документа.

    «A216 WCB» → «ASTM A216 WCB»: пункт про порядок написания спецификации модель читает как
    разрешение дописать издающий орган, которого в записи нет. Убираем только если после этого
    ответ совпадает с записью по буквам и цифрам — значит всё остальное осталось на месте.
    Замена значения («Cu» → «Copper») и дописанная скобка с пересчётом под это не подходят.
    """
    value_words = {w.lower() for w in _WORD_RE.findall(str(value))}
    candidates = [w for w in dict.fromkeys(_WORD_RE.findall(str(answer)))
                  if w.isalpha() and w.lower() not in value_words and w.lower() not in allowed]
    if not candidates:
        return answer

    result = str(answer)
    for word in candidates:
        result = re.sub(rf"\b{re.escape(word)}\b", "", result)
    result = re.sub(r"\s{2,}", " ", result).strip()
    if _alnum(result) != _alnum(value):
        return answer  # убрали слова — и ответ перестал быть этой записью: значит не лишние
    logger.info("  [ADDED] %r -> %r: убраны слова, которых нет в записи (%s)",
                answer, result, ", ".join(candidates))
    return result


def _relabel_sanctioned(value: str, answer: str, rules_text: str) -> bool:
    """Показывает ли документ ту же смену единицы при том же числе своим примером.

    Разность температур пишется в K тем же числом, что и в °C («5 °C» → «5 K»): это не подмена,
    а правило, и пример документа говорит об этом прямо.
    """
    v = _NUM_UNIT_RE.fullmatch(str(value))
    a = _NUM_UNIT_RE.fullmatch(str(answer))
    if not v or not a:
        return False
    for source, target in _EXAMPLE_RE.findall(str(rules_text)):
        s = _NUM_UNIT_RE.fullmatch(source.strip())
        t = _NUM_UNIT_RE.fullmatch(target.strip())
        if (s and t and _unit_key(s.group(3)) == _unit_key(v.group(3))
                and _unit_key(t.group(3)) == _unit_key(a.group(3))
                and s.group(1).replace(",", ".") == t.group(1).replace(",", ".")):
            return True
    return False


# Обе стороны примеров документа: запись «до» и её нормализованный вид.
_EXAMPLE_CELL_RE = re.compile(
    r"\b(?:recorded|raw|normalized)\s+value\s*:\s*['\"“]([^'\"”]{1,120})['\"”]", re.I)
_EXAMPLE_RECORD_RE = re.compile(r"\b(?:raw|normalized)\s+record\b[^:]*:\s*(.+)$", re.I)
_BARE_ARROW_RE = re.compile(r"^\s*(?:\[[^\]]*\]\s*)?([^\"“'→]{1,80}?)\s*(?:->|→)\s*([^\"“'→]{1,80})$")


def _example_texts(rules_text: str) -> list[str]:
    """Текст примеров документа, обе стороны: EXAMPLE, образец датащита, worked example, «X → Y».

    Смотреть только на запись «до» мало: «Cu» превращалось в «Aluminium» — результат примера
    «Al → Aluminium», слова из исходной стороны там нет вовсе.
    """
    texts = [part for pair in _EXAMPLE_RE.findall(str(rules_text)) for part in pair]
    for line in str(rules_text).splitlines():
        texts += _EXAMPLE_CELL_RE.findall(line)
        record = _EXAMPLE_RECORD_RE.search(line)
        if record:
            texts.append(record.group(1))
        bare = _BARE_ARROW_RE.match(line)
        if bare and not _EXAMPLE_RE.search(line):
            texts += [bare.group(1), bare.group(2)]
    return texts


def _example_leak(value: str, answer: str, rules_text: str, allowed: set = frozenset()) -> list[str]:
    """Слова ответа, взятые из записи-примера документа, а не из этой записи.

    «Dependent, pressure» превращалось в «Pressure independent»: пример документа —
    «Independent, pressure» → «Pressure independent», и модель перенесла из него слово. Слово
    из исходной стороны примера — это содержимое чужой записи; в ответ оно может попасть, только
    если есть и в этой записи. Исключение — пример, чья исходная запись целиком входит в эту
    («Al» → «Aluminium» для записи «Al»): тогда его результат и есть предписанный ответ.
    """
    # слова — буквенные цепочки: в «DN32» слово «DN» есть, и «DN 32» из записи не заимствует
    words = lambda text: {w.lower() for w in re.findall(r"[^\W\d_]+", str(text)) if len(w) > 1}
    value_words = words(value)
    licensed: set = set()
    for left, right in _EXAMPLE_RE.findall(str(rules_text)):
        if words(left) and words(left) <= value_words:
            licensed |= words(right)
    foreign = set().union(*[words(s) for s in _example_texts(rules_text)]) if rules_text else set()
    # единица («in» против записанного «inch») — не содержимое чужой записи; pint различает
    # регистр («kPa», но не «kpa»), поэтому проверяется написание из самого ответа
    written = {w.lower(): w for w in re.findall(r"[^\W\d_]+", str(answer))}
    # код или корень, который документ велит подставить («CCW-OPEN», «XB» в «XB193B7»), пишется
    # заглавными так же, как в правилах: это обозначение документа, а не содержимое чужой записи
    codes = {w for w in re.findall(r"[^\W\d_]+", str(rules_text)) if len(w) > 1 and w.isupper()}
    return sorted(w for w in words(answer) - value_words
                  if w in foreign and w not in licensed and w not in allowed
                  and not is_unit(written.get(w, w)) and written.get(w, w) not in codes)


def _rejection_reason(attribute: str, value: str, answer: str, rule_set,
                      rules_text: str = "") -> str:
    """Почему ответ нельзя принять: одна проверка вместо четырёх подряд у вызывающего.

    Каждая из них появилась из конкретной поломки, но снаружи это один вопрос — обоснован ли
    ответ правилами документа. Пустая строка значит «принимаем как есть».
    """
    if _unit_invented(attribute, value, answer):
        return "единицы нет ни в записи, ни в подписи — оставлено как записано"
    if _root_misapplied(answer, rule_set, value):
        return "корень обозначения навешен не на его номер — оставлено как записано"
    if (_unit_relabelled(attribute, value, answer)
            and not _relabel_sanctioned(value, answer, rules_text)):
        return "единица подменена без пересчёта — оставлено как записано"
    if _root_invented(value, answer, rule_set, attribute):
        return "запись не называет это обозначение — оставлено как записано"
    leak = _example_leak(value, answer, rules_text, _document_word_tokens(rule_set, attribute))
    if leak:
        return ("в ответе слово из примера документа, которого нет в записи («"
                + "», «".join(leak) + "») — оставлено как записано")
    return ""


def _partial_separator_edit(original: str, answer: str) -> str | None:
    """Находит правку, применённую к части одинаковых символов вместо всех.

    Правило про разделители действует на всё значение целиком: «C-Mn-Si» → «C-MnSi» (убран один
    дефис из двух) — это не прочтение правила, а промах. Сравнение имеет смысл только пока
    содержимое значения то же самое; если оно заменено кодом или пересобрано, символы исходной
    записи сравнивать не с чем.
    """
    original, answer = str(original), str(answer)
    if _char_bag(original) != _char_bag(answer):
        return None
    for character in sorted(set(re.findall(r"[^\w\s]", original))):
        before, after = original.count(character), answer.count(character)
        if 0 < after < before:
            return f"{character!r}: было {before}, осталось {after}"
    return None


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
    равенстве — более полное: сокращать то, что документ написал с точкой, мы не вправе.
    """
    groups: dict[tuple, list[int]] = {}
    for i, row in enumerate(rows):
        m = _NUM_UNIT_RE.fullmatch(str(row["Rules_Value"]))
        if not m or not unit_mismatch(m.group(3), "count"):
            continue  # не единица измерения — сводить написание нечего
        key = (_norm_attr(row["Attribute"]), _norm_attr(row.get("Context", "")),
               _unit_key(m.group(3)))
        groups.setdefault(key, []).append(i)

    for indices in groups.values():
        spellings = [_NUM_UNIT_RE.fullmatch(str(rows[i]["Rules_Value"])).group(3)
                     for i in indices]
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
            m = _NUM_UNIT_RE.fullmatch(str(rows[i]["Rules_Value"]))
            fixed = f"{m.group(1)}{m.group(2)}{canonical}"
            if fixed != str(rows[i]["Rules_Value"]):
                _note_row(rows[i], f"написание единицы сведено к «{canonical.strip()}» "
                                   f"(как в остальных строках этого атрибута)")
            rows[i]["Rules_Value"] = fixed
    return rows


FUZZY_THRESHOLD = 90
# Порог «похоже хоть на что-то»: ниже него реестр даже не обсуждается с моделью.
REGISTER_FLOOR = 40
# Формулировки исходов (легко локализовать/поменять)
PENDING_WORDING = "pending registration"
AMBIGUOUS_WORDING = "ambiguous match"


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
                          model=None, attribute: str = "") -> tuple[str, bool, str] | None:
    """Сопоставляет значение с одним закрытым регистром «фраза таблицы -> код».

    Возвращает (результат, совпадение точное по составу слов). Точное/почти точное совпадение
    → код; серая зона отдаётся модели-арбитру (выбор только из реестра); «не похоже ни на что»
    → pending registration; «похоже сразу на два пункта» → ambiguous match. К исходу не
    приписывается код: раньше это был префикс ближайшего пункта, и ячейка «THRD pending
    registration: Wafer-type, lugged» читалась как присуждённый код совсем другого соединения.
    """
    if not table:
        return None
    # запись уже является кодом этого реестра — она нормализована, искать нечего
    stripped = str(value).strip()
    for code in table.values():
        if stripped.lower() == str(code).strip().lower():
            return str(code), True, f"запись уже код реестра ({code})"
    norm_value = _normalize_phrase(value)
    scored = sorted(
        ((fuzz.token_sort_ratio(norm_value, _normalize_phrase(key)), key) for key in table),
        reverse=True,
    )
    best_score, best_key = scored[0]
    if best_score >= threshold:
        if _wording_matches(value, best_key):
            return table[best_key], True, f"реестр: «{best_key}» → {table[best_key]}"
        # похоже на пункт реестра, но формулировка другая — записываем как есть, не кодируем
        logger.info("  [REGISTER] %r ~ %r, формулировка не совпадает: оставляем как записано",
                    value, best_key)
        return (value.strip(), False,
                f"реестр: ближайший пункт «{best_key}», формулировка не совпадает — не кодируем")

    if best_score >= REGISTER_FLOOR:
        verdict = _adjudicate_register(model, attribute, value, table)
        if verdict and verdict not in ("AMBIGUOUS", "UNMATCHED"):
            # код присуждается, только если у него есть пункт с тем же составом слов;
            # «по смыслу похоже» — это приближение, а приближать закрытый реестр нельзя:
            # запись остаётся как есть и уходит к ревьюеру
            if any(code == verdict and _wording_matches(value, phrase)
                   for phrase, code in table.items()):
                return verdict, True, f"реестр (выбор модели): {verdict}"
            logger.info("  [REGISTER] %r ~ %r по смыслу, формулировка не совпадает: "
                        "оставляем как записано", value, verdict)
            return (value.strip(), False,
                    f"реестр: по смыслу {verdict}, но формулировка не совпадает — не кодируем")
        if verdict == "AMBIGUOUS":
            return (f"{AMBIGUOUS_WORDING}: {value.strip()}", False,
                    "реестр: значение подходит сразу под несколько пунктов")
        if verdict == "UNMATCHED":
            return (f"{PENDING_WORDING}: {value.strip()}", False,
                    "реестр: такого пункта в документе нет")
    return (f"{PENDING_WORDING}: {value.strip()}", False,
            "реестр: такого пункта в документе нет")


def _names_token(attribute: str, token: str) -> bool:
    """Назван ли этот keyword/root отдельным словом в подписи атрибута."""
    token = str(token).strip().lower()
    if len(token) < 2:
        return False
    return token in re.findall(r"[a-z0-9<>]+", str(attribute).lower())


def _registers_for_attribute(rule_set, attribute: str) -> list[str]:
    """Префиксы реестров, относящихся к атрибуту строки.

    Имя группы правил в документе и подпись атрибута в таблице заказчика — разные тексты
    («nominal diameter (metric)» против «Nominal size, DN»), поэтому точного совпадения ключа
    мало: доискиваем по вхождению, как и для владельцев keyword-правил.
    """
    exact = rule_set.attribute_to_register.get(attribute.strip().lower())
    if exact:
        return exact
    for owner, prefixes in rule_set.attribute_to_register.items():
        if _attribute_matches(attribute, owner):
            return prefixes
    return []


def apply_deterministic(value: str, rule_set, attribute: str,
                        model=None) -> tuple[str, bool, str] | None:
    """Детерминированные преобразования.

    Возвращает (значение, нужна ли ручная проверка, чем закрыто), если строка полностью
    закрыта детерминированными/фразовыми правилами; иначе None (идём в LLM).
    """
    result = value.strip()

    # 0) атрибут исключён из реформатирования — возвращаем без изменений, LLM не вызываем.
    if _attribute_is_excluded(attribute, rule_set.excluded_attributes):
        return result, False, "атрибут исключён из переформатирования"

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
            return "; ".join(mapped), False, "таблица фраз документа"

    # 2) keyword → root: всё значение целиком должно быть «keyword число» / «число keyword».
    # Идёт раньше реестров: «Connector size 18» — это обозначение, а не пункт таблицы
    # типов коннекторов, к которой его притягивает нечёткое совпадение имени атрибута.
    for keyword in sorted(rule_set.deterministic, key=len, reverse=True):
        root = rule_set.deterministic[keyword]
        # attribute-aware: применяем keyword только к его атрибуту-владельцу. Имя группы в
        # документе может не совпадать с подписью атрибута в таблице, поэтому засчитываем и
        # случай, когда сам keyword или его root назван в подписи («Nominal size, DN» ↔ DN).
        owner = rule_set.deterministic_attributes.get(keyword, "")
        if not (_attribute_matches(attribute, owner)
                or _names_token(attribute, keyword) or _names_token(attribute, root)):
            continue
        # В таблице стандарта keyword назван полностью («Pressure class»), а в датащите пишут
        # его коротко («Class 600»), поэтому значащие слова keyword'а тоже считаются его формой.
        forms = {keyword, root} | {w for w in re.findall(r"[A-Za-z]{4,}", keyword)}
        token = "(?:" + "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True)) + ")"
        num = r"(\d+[A-Za-z]*)"
        m = (re.fullmatch(rf"\s*{token}\s*{num}\s*", result, re.IGNORECASE)
             or re.fullmatch(rf"\s*{num}\s*{token}\s*", result, re.IGNORECASE))
        if m:
            num_part = m.group(1)
            nf = rule_set.deterministic_order.get(keyword, rule_set.number_first)
            return (((num_part + root) if nf else (root + num_part)), False,
                    f"корень обозначения: «{keyword}» → {root}")

    # 3) сопоставление с закрытыми регистрами, соответствующими атрибуту.
    prefixes = _registers_for_attribute(rule_set, attribute)
    if prefixes:
        table: dict[str, str] = {}
        for p in prefixes:
            table.update(rule_set.closed_registers.get(p, {}))
        if table:
            # значение через ';' — перечисление независимых позиций: каждая ищется отдельно,
            # иначе целиком оно ни одному пункту реестра не соответствует по определению
            mapped, needs_review, verdict = [], False, ""
            notes: list[str] = []
            for part in (p.strip() for p in result.split(";")):
                if not part:
                    continue
                matched = match_closed_register(part, table, model=model, attribute=attribute)
                if matched is None:
                    break
                code, exact, note = matched
                notes.append(note)
                if _is_pending_result(code):
                    # одна позиция перечисления не разобрана: склеивать её исход с кодами
                    # остальных нельзя — ячейка перестаёт быть и значением, и сообщением
                    verdict = (AMBIGUOUS_WORDING if AMBIGUOUS_WORDING in code.lower()
                               else PENDING_WORDING)
                    break
                mapped.append(code)
                needs_review = needs_review or not exact
            if verdict:
                return f"{verdict}: {result}", True, "; ".join(dict.fromkeys(notes))
            if mapped:
                return "; ".join(mapped), needs_review, "; ".join(dict.fromkeys(notes))

    return None


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


def start_row(state: GraphState, config: RunnableConfig) -> GraphState:

    update: dict = {}
    data_frame = state["data_frame"]
    rule_set = config["configurable"]["rule_set"]
    model = config["configurable"]["model"]

    # --- фиксируем результат ПРЕДЫДУЩЕЙ строки ---
    if state["index_row_table"] != 0:
        # needs_review — только пометка для ревьюера (жёлтая заливка), значение не подменяет
        needs_review = False
        applied: list[str] = [n for n in [state.get("applied", "")] if n]
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
            # смешение предметов в найденных правилах — тоже повод показать строку человеку
            # правила чужих изделий важны, только если они что-то изменили: на нетронутой
            # записи это не «спорно», а просто «в этом документе для неё правил нет»
            changed = str(last_answer).strip() != str(state["original_value"]).strip()
            needs_review = (is_ambiguous or state.get("needs_review_flag", False)
                            or (state.get("subject_conflict", False) and changed))
            rules_value = state["original_value"] if is_ambiguous else last_answer
            if not is_ambiguous:
                rules_value = _strip_redundant_parenthetical(rules_value, state["original_value"])
                # финальная строка собирается детерминированным форматтером, если для атрибута
                # из документа извлечён жёсткий шаблон рендера (знаки после запятой + суффикс)
                template = _match_render_template(state["attribute"], rule_set.render_templates,
                                                  state.get("rule_group_names"), rules_value)
                if template:
                    rendered = _apply_render_template(rules_value, template)
                    if rendered != rules_value:
                        applied.append(f"шаблон рендера документа ({template['decimals']} зн., "
                                       f"«{template['suffix'].strip()}»)")
                    rules_value = rendered
                # проходы идут один за другим: каждый следующий видит результат предыдущего
                aligned = _align_root_spacing(rules_value, rule_set,
                                              state.get("rules_text", ""))
                if aligned != rules_value:
                    applied.append("написание корня выровнено по документу")
                rules_value = aligned
                # если документ прописал пробел, формат записи ему не указ: правило главнее
                if not _document_mandates_spacing(state.get("rules_text", "")):
                    spaced = _restore_unit_spacing(state["original_value"], rules_value)
                    if spaced != rules_value:
                        applied.append("пробел у единицы взят из записи")
                    rules_value = spaced
                stripped_words = _strip_added_words(
                    state["original_value"], rules_value,
                    _document_word_tokens(rule_set, state["attribute"],
                                          state.get("rule_group_names"))
                    | _mandated_additions(state.get("rules_text", "")))
                if stripped_words != rules_value:
                    applied.append("убраны слова, которых нет в записи")
                rules_value = stripped_words
                guard = _rejection_reason(state["attribute"], state["original_value"],
                                          rules_value, rule_set, state.get("rules_text", ""))
                if guard:
                    logger.info("  [GUARD] %r -> %r: %s",
                                state["original_value"], rules_value, guard)
                    rules_value = state["original_value"]
                    needs_review = True
                    applied.append(guard)
                # правка разделителей, применённая к части вхождений, — повод показать строку
                partial = _partial_separator_edit(state["original_value"], rules_value)
                if partial:
                    logger.info("  [EDIT] %r -> %r: правка не по всем вхождениям (%s)",
                                state["original_value"], rules_value, partial)
                    needs_review = True
                    applied.append(f"правка разделителя не по всем вхождениям ({partial})")
            if is_ambiguous:
                applied.append("правила документа дают два разных ответа")
            elif state.get("needs_review_flag"):
                applied.append("прогоны разошлись между собой")
            if any("ответ модели не получен" in n for n in applied):
                needs_review = True  # строка осталась как записана не по правилу, а по сбою
            if not changed:
                missing = _unresolved_references(
                    rule_set, _own_rules_text(rule_set, state["attribute"],
                                              state.get("rule_group_names"),
                                              state.get("rules_text", "")))
                if missing:
                    logger.info("  [REFS] неразрешимые ссылки: %s", missing)
                    needs_review = True
                    applied.append("правило ссылается на пункт " + ", ".join(missing)
                                   + " — его текста нет в базе")
            if state.get("subject_conflict"):
                applied.append("найдены правила разных изделий, контекст строки не уточняет"
                               if changed else
                               "правила этого документа относятся к другим изделиям")
        else:
            rules_value = state["original_value"]
        # что именно изменилось — видно из самой строки, и это честнее пересказа модели
        change = _describe_change(state["original_value"], rules_value)
        if change and not any("пересчёт единиц" in n for n in applied):
            applied.insert(0, change)

        # строка изменилась, а сослаться не на что: это не «правил нет», это повод посмотреть
        if not [n for n in applied if n] and _alnum(rules_value) != _alnum(state["original_value"]):
            applied.append("значение изменено, но правило не названо")
            needs_review = True

        new_row = {
            "Attribute": state["attribute"],
            "Value": state["original_value"],
            "Rules_Value": rules_value,
            "Ambiguous": needs_review,
            "Context": state.get("row_context", ""),
            "Deterministic": state.get("deterministic_result") is not None,
            "Applied": ("; ".join(dict.fromkeys(n for n in applied if n))
                        or "правила этого документа к строке не применялись"),
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
        update["applied"] = ""
        update["messages"] = [RemoveMessage(id=k.id) for k in state["messages"]]

        if not _document_covers_row(rule_set, context, model):
            logger.info("=" * 80)
            logger.info("Row %d/%d | Attribute: %r | %r — документ не про это изделие, пусто",
                        state["index_row_table"] + 1, data_frame.shape[0], attribute, context)
            # пустая ячейка сама по себе говорит «документ не про это изделие» —
            # заливать её жёлтым не за чем: ревьюеру там нечего решать
            update["deterministic_result"] = ""
            update["needs_review_flag"] = False
            update["applied"] = "документ не распространяется на это изделие"
            update["value"] = ""
            update["rules_text"] = ""
            update["rule_group_names"] = []
            update["subject_conflict"] = False
            return update

        det = apply_deterministic(value, rule_set, attribute, model=model)
        if det is not None:
            det_value, det_needs_review, det_note = det
            update["applied"] = det_note
            # полностью детерминировано — общий LLM-путь не нужен
            update["deterministic_result"] = det_value
            update["needs_review_flag"] = det_needs_review
            update["value"] = det_value
            update["rules_text"] = ""
            update["rule_group_names"] = []
            update["subject_conflict"] = False
            logger.info("=" * 80)
            logger.info(
                "Row %d/%d | Attribute: %r | Value: %r | DETERMINISTIC -> %r",
                state["index_row_table"] + 1, data_frame.shape[0], attribute, value, det_value,
            )
        else:
            update["deterministic_result"] = None
            update["value"] = value
            (update["rules_text"], update["rule_group_names"],
             update["subject_conflict"]) = build_rules_text(rule_set, attribute, value, context,
                                                            model=model)
            logger.info("=" * 80)
            logger.info(
                "Row %d/%d | Attribute: %r | Value: %r | rules: %d chars",
                state["index_row_table"] + 1, data_frame.shape[0], attribute, value,
                len(update["rules_text"]),
            )

    return update


def _tool_note(name: str, args: dict, result: str) -> str:
    """Что именно сделал инструмент — словами, по его же аргументам и ответу.

    Вызов, ничего не изменивший (замена строки на саму себя, пересчёт в ту же единицу),
    в объяснение не идёт: это не применённое правило, а холостой ход.
    """
    if str(result).startswith("cannot convert"):
        return f"пересчёт не выполнен: {result}"
    if name == "convert_units":
        source, target = str(args.get("quantity", "")), str(result)
        source_unit = _NUM_UNIT_RE.fullmatch(source.strip())
        target_unit = _NUM_UNIT_RE.fullmatch(target.strip())
        if (source_unit and target_unit
                and _unit_key(source_unit.group(3)) == _unit_key(target_unit.group(3))):
            return "" if source.strip() == target.strip() else f"округление: «{source}» → {target}"
        return f"пересчёт единиц: «{source}» → {target}"
    if name == "remove_characters":
        return (f"удалены символы «{args.get('characters')}» из «{args.get('text')}»"
                if str(args.get("characters", "")) else "")
    if name == "replace_substring":
        if str(args.get("find")) == str(args.get("replace_with")):
            return ""
        return (f"замена «{args.get('find')}» на «{args.get('replace_with')}» "
                f"в «{args.get('text')}»")
    return f"{name}: {result}"


# Потолок шагов цикла инструментов на один сэмпл. Штатной строке хватает одного-двух
# вызовов; без потолка модель, зациклившаяся на холостой замене, молотит бесконечно.
MAX_TOOL_STEPS = 6
_REPEATED_CALL_REPLY = ("this exact call was already made and returned the same result; "
                        "do not call it again — answer now with the JSON object")
_BUDGET_REPLY = ("no more tool calls are available for this value; answer now with the "
                 "JSON object, using what the previous calls returned")


def _run_tool_call(tc: dict, notes: list[str]) -> str:
    """Выполняет один вызов и копит заметку о сделанном."""
    tool_fn = TOOLS_BY_NAME.get(tc["name"])
    if tool_fn is None:
        return "unknown tool"
    try:
        content = str(tool_fn.invoke(tc["args"]))
    except Exception as e:
        return f"error: {e}"
    logger.info("  [TOOL] %s(%s) -> %r", tc["name"], tc["args"], content)
    note = _tool_note(tc["name"], tc["args"], content)
    if note:
        notes.append(note)
    return content


def _invoke_with_tools(model, messages) -> tuple:
    """Инвокает модель, прогоняя цикл tool-calls. Возвращает (ответ, что сделали инструменты).

    Повторный вызов с теми же аргументами не выполняется, а число шагов ограничено: иначе
    модель, которой инструмент возвращает то же самое, вызывает его снова и снова.
    """
    notes: list[str] = []
    seen: set = set()
    response = model.invoke(messages)

    for _ in range(MAX_TOOL_STEPS):
        calls = getattr(response, "tool_calls", None)
        if not calls:
            return response, notes
        tool_messages = []
        for tc in calls:
            key = (tc["name"], json.dumps(tc["args"], sort_keys=True, default=str))
            if key in seen:
                logger.info("  [TOOL] повтор %s(%s) — не выполняем", tc["name"], tc["args"])
                content = _REPEATED_CALL_REPLY
            else:
                seen.add(key)
                content = _run_tool_call(tc, notes)
            tool_messages.append(ToolMessage(content=content, tool_call_id=tc["id"]))
        messages = messages + [response] + tool_messages
        response = model.invoke(messages)

    if getattr(response, "tool_calls", None):
        logger.warning("  [TOOL] исчерпан лимит вызовов (%d), просим ответить без инструментов",
                       MAX_TOOL_STEPS)
        closing = [ToolMessage(content=_BUDGET_REPLY, tool_call_id=tc["id"])
                   for tc in response.tool_calls]
        messages = messages + [response] + closing
        response = model.invoke(messages)
    return response, notes


def _normalize_once(model, attribute: str, value: str, rules_text: str,
                    context: str = "") -> tuple[str, str]:
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
    response, tool_notes = _invoke_with_tools(model, messages)
    payload = _parse_payload(response.content)
    candidate = _parse_result(json.dumps(payload) if payload else "").strip()
    applied = "; ".join(x for x in (*tool_notes, _cited_clauses(payload)) if x)
    if not candidate:
        # модель не дала разбираемого ответа (например, упёрлась в лимит вызовов):
        # запись остаётся как есть, а строка получает пометку
        logger.warning("  [NORMALIZE] ответа нет, оставляем запись %r", value)
        return str(value), "; ".join(x for x in (applied, "ответ модели не получен") if x)
    if not candidate or _is_ambiguous_answer(candidate):
        return candidate, applied

    # второй проход: precondition, исключения, утечка из worked-example, полнота
    messages = [
        {"role": "system", "content": VERIFY_PROMPT},
        {"role": "user", "content": (f"Attribute: {attribute}\n{_context_line(context)}"
                                     f"Original value: {value}\n"
                                     f"Candidate answer: {candidate}\n\n"
                                     f"Rules:\n{rules_text}")},
    ]
    review_response, review_tools = _invoke_with_tools(model, messages)
    review = _parse_payload(review_response.content)
    checked = str(review.get("result", "")).strip()
    if checked and checked != candidate:
        logger.info("  [VERIFY] %r -> %r", candidate, checked)
        extra = "; ".join(x for x in (*review_tools, _cited_clauses(review)) if x)
        applied = "; ".join(x for x in (applied, extra) if x)
    return (checked or candidate), applied


def _vote_normalize(model, attribute: str, value: str, rules_text: str,
                    context: str, samples: int, allowed: set = frozenset()) -> tuple[str, bool, str]:
    """Несколько независимых сэмплов и большинство; второй элемент — разошлись ли они.

    Спорные строки модель трактует по-разному от запуска к запуску, поэтому берётся
    несколько сэмплов: это и повышает точность, и делает результат воспроизводимым
    (одна и та же таблица даёт одну и ту же выдачу).
    """
    args = (model, attribute, value, rules_text, context)
    if samples <= 1:
        sampled = [_normalize_once(*args)]
    else:
        with ThreadPoolExecutor(max_workers=samples) as pool:
            sampled = list(pool.map(lambda _: _normalize_once(*args), range(samples)))

    # сэмпл, перенёсший слово из записи-примера, не голосует: иначе при двух таких из трёх
    # побеждал чужой ответ. Если перенесли все, решает guard после голосования
    leaks = [_example_leak(value, v, rules_text, allowed) for v, _ in sampled]
    clean = [s for s, leak in zip(sampled, leaks) if not leak]
    if clean and len(clean) < len(sampled):
        logger.info("  [LEAK] отброшено сэмплов со словами из примеров: %d", len(sampled) - len(clean))
        sampled = clean
    elif not clean and any(leaks):
        # перенесли все: ещё одна попытка, где названо, какое слово чужое. Удалась — ответ
        # берём, но строку помечаем: сама модель с первого раза этого не увидела
        borrowed = sorted(set().union(*map(set, leaks)))
        note = ("\n\nNote on earlier attempts: they put " + ", ".join(f'"{w}"' for w in borrowed)
                + " into the answer. That comes from a worked example of the document, not from "
                  "this value. Apply the rule's mechanism to this value's own words only.")
        retry, retry_applied = _normalize_once(model, attribute, value, rules_text + note, context)
        if not _example_leak(value, retry, rules_text, allowed):
            logger.info("  [LEAK] все сэмплы взяли %s из примеров, повтор дал %r", borrowed, retry)
            return retry, True, retry_applied

    # большинство; при равенстве — вариант, встретившийся раньше (стабильный tie-break)
    candidates = [value for value, _ in sampled]
    counts = Counter(candidates)
    answer = max(candidates, key=lambda c: (counts[c], -candidates.index(c)))
    applied = next((note for value, note in sampled if value == answer), "")
    if len(counts) > 1:
        logger.info("  [VOTE] %r -> %r", dict(counts), answer)
    return answer, len(counts) > 1, applied


def normalize_row(state: GraphState, config: RunnableConfig) -> GraphState:
    """Ответ модели по полному набору правил, устойчивый к разбросу между прогонами."""
    model = config["configurable"]["model"]
    samples = max(1, int(config["configurable"].get("self_consistency", SELF_CONSISTENCY)))
    rule_set = config["configurable"]["rule_set"]
    answer, split, applied = _vote_normalize(model, state["attribute"], state["value"],
                                             state["rules_text"], state.get("row_context", ""),
                                             samples,
                                             _document_word_tokens(rule_set, state["attribute"]))
    logger.info("  [NORMALIZE] %r -> %r", state["value"], answer)

    # разошедшиеся сэмплы = строка, которую правила документа не решают однозначно:
    # значение берём по большинству, но помечаем строку на ручную проверку
    return {"answer": answer, "needs_review_flag": split, "applied": applied}


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


# Колонки, которые отдаёт проход по одному документу: задаются явно, иначе таблица из
# нуля строк приходит вовсе без колонок и валит сборку результата.
ROW_COLUMNS = ["Attribute", "Value", "Rules_Value", "Ambiguous", "Context", "Deterministic",
               "Applied"]


def run_single_rule_set(data_frame: pd.DataFrame, model, rule_set) -> pd.DataFrame:
    initial_state = {
        "messages": [],
        "index_row_table": 0,
        "data_frame": data_frame,
        "rules_text": "",
        "rule_group_names": [],
        "row_context": "",
        "subject_conflict": False,
        "new_rows": [],
        "original_value": "",
        "deterministic_result": None,
        "answer": None,
        "needs_review_flag": False,
        "applied": "",
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
    return pd.DataFrame(_canonicalize_units(_canonicalize_formatting(rows)), columns=ROW_COLUMNS)


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

        row_rules = _adhoc_rules_for_attribute(rules, attribute)
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


# Excel не принимает управляющие символы и обрывается на ячейке длиннее 32767 знаков.
_ILLEGAL_XLSX_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
EXCEL_CELL_LIMIT = 32767


def _excel_safe(value):
    """Значение → то, что Excel примет: без управляющих символов и не длиннее предела."""
    if not isinstance(value, str):
        return value
    cleaned = _ILLEGAL_XLSX_RE.sub("", value)
    return cleaned[:EXCEL_CELL_LIMIT] if len(cleaned) > EXCEL_CELL_LIMIT else cleaned


def _write_excel_with_highlight(df: pd.DataFrame, ambiguous_cols: dict[str, list[bool]]) -> io.BytesIO:
    """Пишет DataFrame в Excel, заливая спорные ячейки жёлтым."""
    from openpyxl import Workbook
    from openpyxl.styles import PatternFill

    wb = Workbook()
    ws = wb.active
    yellow = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")

    ws.append([_excel_safe(c) for c in df.columns])
    for _, row in df.iterrows():
        ws.append([_excel_safe(v) for v in row.tolist()])
    # записанное значение — это данные, а не формула: «=SUM(...)» из датащита должно
    # остаться текстом, а не вычисляться при открытии файла
    for row in ws.iter_rows():
        for cell in row:
            if cell.data_type == "f":
                cell.data_type = "s"

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


XLSX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


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


@app.post("/process")
def process_excel(
        file: UploadFile = File(...),
        app_config: AppConfig = Depends(get_app_config),
):
    contents = file.file.read()
    data_frame = pd.read_excel(io.BytesIO(contents))
    result_df, ambiguous_cols = normalize_table(data_frame, app_config)
    output_buffer = _write_excel_with_highlight(result_df, ambiguous_cols)

    return Response(
        content=output_buffer.read(),
        media_type=XLSX_MEDIA_TYPE,
        headers={"Content-Disposition": "attachment; filename=result.xlsx"},
    )


_LABEL_TAIL_RE = re.compile(r"\s*(?:normalization\s+)?rules$", re.I)


def _display_name(name: str) -> str:
    """Имя папки документа → ярлык для интерфейса: подчёркивания в пробелы, хвост долой."""
    label = _LABEL_TAIL_RE.sub("", name.replace("_", " ")).strip()
    return re.sub(r"\s+", " ", label) or name


@app.get("/standards")
def list_standards(app_config: AppConfig = Depends(get_app_config)):
    """Какие стандарты загружены — фронт даёт из них выбрать."""
    return {"standards": [{"name": rs.name, "label": _display_name(rs.name)}
                          for rs in app_config.rule_sets]}


def _parse_json_list(raw: str) -> list:
    """Список приходит с фронта строкой JSON; его порча не должна ронять запрос."""
    try:
        parsed = json.loads(raw) if raw else []
    except ValueError:
        return []
    return parsed if isinstance(parsed, list) else []


def _selected_standard_names(app_config: AppConfig, raw: str) -> list[str]:
    """Имена стандартов, выбранных в интерфейсе; неизвестные отбрасываем."""
    known = {rs.name for rs in app_config.rule_sets}
    return [n for n in (str(x).strip() for x in _parse_json_list(raw)) if n in known]


def _rule_sets_by_name(app_config: AppConfig, names: list[str]) -> list:
    """Имена → наборы правил. Пустой список (и список из одних неизвестных) — значит все."""
    by_name = {rs.name: rs for rs in app_config.rule_sets}
    chosen = [by_name[n] for n in names if n in by_name]
    return chosen or list(app_config.rule_sets)


def _uploaded(file: UploadFile | None) -> UploadFile | None:
    """Пустое поле формы приходит как файл без имени — это не вложение."""
    return file if file is not None and (file.filename or "").strip() else None


def _download_name(stored) -> str:
    """Имя файла для скачивания: то же, что у файла в рабочей области."""
    name = stored.name.strip() or f"{stored.file_id}.xlsx"
    return name if name.lower().endswith(".xlsx") else f"{name}.xlsx"


@app.post("/chat")
def chat(
        message: str = Form(""),
        history: str = Form("[]"),
        standards: str = Form("[]"),
        session_id: str = Form(""),
        file: UploadFile | None = File(None),
        app_config: AppConfig = Depends(get_app_config),
):
    """Вход чата: сообщение ведёт агент, файлы диалога лежат в его рабочей области.

    Сам агент ничего не нормализует — он выбирает файл, стандарты и правила и зовёт
    пайплайн инструментом. Поэтому «пересчитай тот файл, я забыл правило» работает без
    повторной загрузки: файл никуда не девался, пока жив диалог.
    """
    session = (session_id or "").strip() or uuid4().hex
    workspace = get_workspace(session)
    text = (message or "").strip()
    uploaded = _uploaded(file)

    if not text and uploaded is None:
        return {"reply": "Напишите сообщение или приложите файл .xlsx — обработаю.",
                "session_id": session, "files": []}

    notes: list[str] = []
    if uploaded is not None:
        try:
            frame = pd.read_excel(io.BytesIO(uploaded.file.read()))
        except Exception as e:
            logger.warning("[CHAT] файл %r не прочитан: %s", uploaded.filename, e)
            return {"reply": f"Не удалось прочитать «{uploaded.filename}». Нужен .xlsx с "
                             f"колонками Attribute и Value.",
                    "session_id": session, "files": []}
        stored = workspace.add(name=uploaded.filename, frame=frame)
        columns = ", ".join(str(c) for c in frame.columns)
        notes.append(f"The user has just attached the file {stored.name} "
                     f"(file_id={stored.file_id}), {frame.shape[0]} rows, columns: {columns}.")

    selected = _selected_standard_names(app_config, standards)
    notes.append("Standards picked in the interface: "
                 + (", ".join(selected) if selected else "none, so all of them") + ".")

    produced: list[str] = []
    catalogue = [{"name": rs.name, "label": _display_name(rs.name)} for rs in app_config.rule_sets]
    tools = build_tools(
        workspace, catalogue, selected,
        lambda frame, names, rules: normalize_table(
            frame, app_config, _rule_sets_by_name(app_config, names), rules),
        produced,
    )
    prompt = f"[Context: {' '.join(notes)}]\n\n{text or 'Обработай приложенный файл.'}"
    reply = run_agent(app_config.assistant or app_config.model, tools,
                      history_messages(_parse_json_list(history)), prompt)

    files = []
    for file_id in produced:
        stored = workspace.get(file_id)
        if stored is None:
            continue
        buffer = _write_excel_with_highlight(stored.frame, stored.ambiguous)
        files.append({
            "name": _download_name(stored),
            "media_type": XLSX_MEDIA_TYPE,
            "content_base64": base64.b64encode(buffer.read()).decode("ascii"),
        })

    return {"reply": reply, "session_id": session, "files": files}
