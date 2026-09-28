"""Текст правил для строки: поиск групп, блок класса, пункты и таблицы по ссылкам."""
import re

from rapidfuzz import fuzz

from ..modules.config import hybrid_search, RULE_SET_LIBRARY, document_id, value_shape
from ..modules.tables import table_references
from .common import _attribute_matches, _normalize_phrase, logger
from .subjects import _equipment_kind, _row_class, _select_by_subject
from .deterministic import _registers_for_attribute, _synonym_tables_for_attribute


RETRIEVAL_K = 4


# Сколько групп добавить по форме значения, когда имя атрибута не привело ни к одной.
MAX_SHAPE_GROUPS = 2


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
    row_class = _row_class(rule_set, context, model)
    if row_class:
        # раздел класса изделия строки: какие у него атрибуты и в какой форме пишется каждый
        logger.info("  [CLASS] %r -> правила класса %r", context, row_class)
        blocks.append(f"Rules of the equipment class of this row ({row_class}):\n"
                      + rule_set.class_blocks[row_class])
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
    local_text = "\n".join(body for _, body in pairs)
    expanded = "\n\n".join([text] + expand_references(rule_set, text, attribute, names, str(value))
                           + _table_blocks(rule_set, attribute, local_text, text))
    return expanded, names, subject_conflict


MAX_PULLED_TABLES = 4


MAX_TABLE_ROWS = 60


def _table_blocks(rule_set, attribute: str, local_text: str, text: str) -> list[str]:
    """Таблицы, на которые ссылаются найденные правила атрибута, — цитатой из документа.

    «Written form: Canonical value of Table 12.88» без самой таблицы 12.88 модели ничего не
    говорит, а подтягивание по ссылкам знало только номера пунктов. Ссылки берутся из групп,
    найденных для строки, а не из общих положений: те ссылаются на десятки таблиц разом.
    Таблицы синонимов самого атрибута идут первыми, даже если на них никто не сослался.
    """
    tables = getattr(rule_set, "tables", None) or {}
    if not tables:
        return []
    wanted = _synonym_tables_for_attribute(rule_set, attribute) + table_references(local_text)
    quoted: list[str] = []
    for table_id in dict.fromkeys(wanted):
        table = tables.get(table_id)
        if table is None or (table.title and table.title in text):
            continue  # такой таблицы нет или она уже в тексте правил
        quoted.append("\n".join(table.lines()[:MAX_TABLE_ROWS + 1]))
        if len(quoted) >= MAX_PULLED_TABLES:
            break
    if not quoted:
        return []
    logger.info("  [REFS] подтянуто таблиц по ссылкам: %d", len(quoted))
    return ["Tables that the rules above refer to, quoted from the same document. Where a table "
            "lists other written forms of a value, a recorded value that is one of those forms is "
            "written as the value of that row.\n\n" + "\n\n".join(quoted)]


_CLAUSE_DEF_RE = re.compile(r"^\s*(\d+(?:\.\d+)+)\s")


_CLAUSE_REF_RE = re.compile(r"\b\d+(?:\.\d+)+\b")


# Упоминание таблицы/приложения/реестра — признак, что правило отсылает к списку значений.
_TABLE_REF_RE = re.compile(r"\b(?:table|appendix|register|annex)\b", re.I)


MAX_PULLED_CLAUSES = 12


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
