from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

import json
import re
import logging

from rank_bm25 import BM25Okapi

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from graphs.structured_documents.block_grouping import parse_page_content

from .tables import MAX_FORM_CHARS, DocTable, form_key, load_tables, table_references
from .tools import array_of_tools

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv()

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    """Статические настройки приложения."""
    base_dir: Path = BASE_DIR
    rule_groups_dir: Path = BASE_DIR.parent / "structured_documents" / "data" / "output"
    vector_store_dir: Path = BASE_DIR / "data" / "input"

    chat_model: str = "gpt-5.6-luna"
    embedding_model: str = "text-embedding-3-large"
    temperature: float = 0.0
    # Для reasoning-моделей (gpt-5.6-luna) тулзы работают только при reasoning_effort="none".
    reasoning_effort: str | None = "none"
    # Паспортам и сборке правил инструменты не нужны, и рассуждать им можно: на AVS / HB-15
    # (2026-10-13) none — 88 и 89 из 102, low — 96 и 97, при +10–20% времени и ~1.7× выходных токенов.
    spec_reasoning_effort: str | None = "low"


@dataclass
class GroupEntry:
    """Одна группа правил: имя атрибута, алиасы, текст правил и родительская тема."""
    attribute: str
    aliases: list[str]
    body: str
    parent: str = ""
    shapes: set[str] = field(default_factory=set)


_SHAPE_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")


def value_shape(text: str) -> str:
    """Форма числового значения: числа → «9», без пробелов и регистра.

    «600 x 600mm» и «500 x 1200mm» — одна форма «9x9mm», «2.5 bar» и «1.5 bar» — «9bar».
    У значения без цифр формы нет: по словам совпадает слишком многое. Формы групп берутся
    из примеров их карточек (rule_cards.json), а не из стрелок, найденных в тексте регуляркой.
    """
    text = str(text).strip().lower()
    if not re.search(r"\d", text):
        return ""
    return re.sub(r"\s+", "", _SHAPE_NUMBER_RE.sub("9", text))


@dataclass
class RuleSet:
    """Правила одного rule-файла: semantic-индекс + BM25 + список групп."""
    name: str
    global_rules: str
    store: InMemoryVectorStore
    bm25: "BM25Okapi"
    groups: list[GroupEntry]
    deterministic: dict[str, str] = field(default_factory=dict)
    phrase_map: dict[str, str] = field(default_factory=dict)
    closed_registers: dict[str, dict[str, str]] = field(default_factory=dict)
    attribute_to_register: dict[str, list[str]] = field(default_factory=dict)
    number_first: bool = False
    deterministic_order: dict[str, bool] = field(default_factory=dict)
    deterministic_attributes: dict[str, str] = field(default_factory=dict)
    excluded_attributes: list[str] = field(default_factory=list)
    render_templates: dict[str, dict] = field(default_factory=dict)
    parent_to_indices: dict[str, list[int]] = field(default_factory=dict)
    clause_index: dict[str, str] = field(default_factory=dict)
    group_subjects: dict[str, str] = field(default_factory=dict)
    # таблицы документа по номеру и таблицы синонимов «принятая форма → каноническое значение»
    tables: dict[str, DocTable] = field(default_factory=dict)
    synonyms: dict[str, dict[str, str]] = field(default_factory=dict)
    synonym_owners: dict[str, list[str]] = field(default_factory=dict)
    universal_synonyms: list[str] = field(default_factory=list)
    # класс изделия → его правила (определение, таблица атрибутов, пункты раздела класса)
    class_blocks: dict[str, str] = field(default_factory=dict)
    # карточки правил (rule_cards.json): однозначный пересказ групп и общих положений,
    # и принятые формы из них, годные для детерминированной замены (группа → форма → запись)
    cards: dict[str, dict] = field(default_factory=dict)
    global_card: dict = field(default_factory=dict)
    card_forms: dict[str, dict[str, str]] = field(default_factory=dict)
    # примеры «было → стало» из карточек: [{"from", "to", "group"}]
    examples: list[dict] = field(default_factory=list)
    # начало документа как распознано (заголовок, тема, вводная часть): по нему решается,
    # про какое оборудование документ; группировка часть этого в правила не берёт
    opening_text: str = ""


@dataclass
class AppConfig:
    settings: Settings
    model: ChatOpenAI
    rule_sets: list[RuleSet]
    # Та же модель без bind_tools: разговорной части тулзы конверсии единиц не нужны.
    assistant: ChatOpenAI | None = None
    # Модель паспортов (нормализация по спецификации, сборка правил пользователя): с рассуждением.
    spec_model: ChatOpenAI | None = None


# Attribute names that mark document front/back matter rather than technical rules.
_NON_RULE_ATTRIBUTE_MARKERS = (
    "metadata",
    "copyright",
    "legal",
    "liability",
    "disclaimer",
    "table of contents",
    "bibliography",
    "foreword",
    "acknowledgment",
    "contact information",
    "distribution",
    "publisher",
    "provided by",
)


def _is_non_rule(header: str) -> bool:
    return any(marker in header.lower() for marker in _NON_RULE_ATTRIBUTE_MARKERS)


def _split_groups(groups: list[dict]) -> tuple[str, list[GroupEntry]]:
    """Разбивает группы одного документа на (global_rules_text, list[GroupEntry]).

    global — правила, действующие на все атрибуты.
    GroupEntry — local-группы с именем, алиасами и текстом правил (поиск по атрибуту).
    Группы класса изделия (kind="class") сюда не входят — см. _class_blocks.
    """
    global_rules: list[str] = []
    entries: list[GroupEntry] = []

    for group in groups:
        header = group.get("attribute", "").strip()
        body = "\n".join(rule.strip() for rule in group.get("rules", []))

        if group.get("type") == "global":
            global_rules.append(body)
        elif group.get("kind") == "class":
            continue
        else:
            if _is_non_rule(header):
                continue
            # правила идентичности изделия («какие атрибуты делают две записи одним изделием»)
            # называют те же атрибуты, что и правила записи значения, и отъедали у них места
            # в выдаче ретрива, ничего не говоря о том, как значение записать
            if group.get("kind") == "identity":
                continue
            aliases = [a.strip() for a in group.get("aliases", []) if a and a.strip()]
            entries.append(GroupEntry(attribute=header, aliases=aliases, body=body,
                                      parent=group.get("parent", "") or header))

    return "\n".join(global_rules), entries


def _load_rule_cards(doc_dir: Path) -> tuple[dict, dict[str, dict], dict[str, dict[str, str]]]:
    """(общая карточка, карточки групп, принятые формы групп) из rule_cards.json, если он есть.

    Для детерминированной замены берутся только пары, чья запись входит в перечень значений
    самой карточки: карточку пишет модель, и пара «Cv → the number only» — описание шага, а
    не значение. Такие пары остаются в карточке для модели, но вслепую не применяются.
    """
    path = doc_dir / "rule_cards.json"
    if not path.exists():
        return {}, {}, {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}, {}, {}
    cards = data.get("groups") or {}
    forms: dict[str, dict[str, str]] = {}
    for name, card in cards.items():
        values = {form_key(v): str(v).strip() for v in card.get("values") or [] if str(v).strip()}
        mapping: dict[str, str] = {}
        for pair in card.get("accepted") or []:
            written = values.get(form_key(pair.get("written", "")))
            if written:
                mapping.setdefault(form_key(pair.get("form", "")), written)
        # сама запись в другом регистре — тоже она: «seamless» → «Seamless»
        for key, written in values.items():
            mapping.setdefault(key, written)
        if mapping:
            forms[name.strip().lower()] = mapping
    return data.get("global") or {}, cards, forms


def _class_blocks(groups: list[dict]) -> dict[str, str]:
    """Название класса изделия → его правила (группы kind="class").

    Документ на много классов оборудования описывает каждый класс своим разделом: что это,
    какие у него атрибуты и в какой форме пишется каждый. Это правила «на все атрибуты» — но
    одного класса, и в Global им не место: там строка задвижки получала описания всех
    пятидесяти классов. Блок класса в поиск по похожести не идёт — его подмешивают по классу
    строки (api._row_class).
    """
    blocks: dict[str, str] = {}
    for group in groups:
        if group.get("kind") == "class" and group.get("type") != "global":
            name = str(group.get("attribute", "")).strip()
            body = "\n".join(str(r).strip() for r in group.get("rules", []))
            if name and body:
                blocks[name] = (blocks[name] + "\n" + body) if name in blocks else body
    return blocks


# Строка правила, начинающаяся с номера пункта, — определение этого пункта.
_CLAUSE_DEF_RE = re.compile(r"^\s*(\d+(?:\.\d+)+)\s")


def _build_clause_index(global_rules: str, entries: list[GroupEntry]) -> dict[str, str]:
    """Номер пункта → его текст, по всему документу.

    Правила постоянно ссылаются друг на друга («см. 13.1», «под Clause 10»), а в промпт строки
    попадают только global-правила и несколько найденных групп. Индекс позволяет дотянуть
    текст пункта, на который ссылаются, если сам он в выборку не попал.
    """
    index: dict[str, str] = {}
    for body in [global_rules] + [e.body for e in entries]:
        for line in body.splitlines():
            match = _CLAUSE_DEF_RE.match(line)
            if match:
                index.setdefault(match.group(1), line.strip())
    return index


_ATTACHED_RE = re.compile(r"^\s*(?:EXAMPLE|NOTE)\b", re.I)


def _clause_index_from_pages(doc_dir: Path) -> dict[str, str]:
    """Номер пункта → текст пункта (с его EXAMPLE/NOTE), прямо по тексту страниц документа.

    Индекс по группам зависит от того, сохранила ли группировка номера пунктов, а она их
    теряла массово (в иных документах ни одного): ссылка «see 6.2» тогда вела в никуда, хотя
    сам пункт извлечён. Текст страниц (data_after_check.json) номера хранит всегда.
    """
    path = doc_dir / "data_after_check.json"
    if not path.exists():
        return {}
    try:
        pages = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    index: dict[str, str] = {}
    last = ""
    for page in pages:
        content = page.get("content") if isinstance(page, dict) else None
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                continue
        for block in (content or {}).get("blocks", []):
            if block.get("type") != "paragraph":
                continue
            text = str(block.get("text", "")).strip()
            match = _CLAUSE_DEF_RE.match(text)
            if match:
                last = match.group(1)
                index.setdefault(last, text)
            elif last and _ATTACHED_RE.match(text) and last in index:
                index[last] += "\n" + text  # пример пункта, даже если он на следующей странице
    return index


def _search_text(entry: GroupEntry) -> str:
    """Текст, который эмбедится и по которому строится BM25: имя + алиасы."""
    return " | ".join([entry.attribute] + entry.aliases)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def load_vector_store(settings: Settings, entries: list[GroupEntry], name: str) -> InMemoryVectorStore:
    """Строит (или грузит из кэша) semantic-индекс: эмбедится имя + алиасы."""
    embeddings = OpenAIEmbeddings(model=settings.embedding_model)
    cache_path = settings.vector_store_dir / f"vector_store_{name}.json"

    if cache_path.exists():
        return InMemoryVectorStore.load(str(cache_path), embeddings)

    store = InMemoryVectorStore(embeddings)
    docs = [
        Document(page_content=_search_text(e), metadata={"text": e.body, "index": i})
        for i, e in enumerate(entries)
    ]
    store.store.clear()
    store.add_documents(docs)
    store.dump(str(cache_path))
    return store


# Все загруженные документы по их обозначению («OGXSTD410»): правило одного документа может
# ссылаться на пункт другого («per 6.2 of OGX-STD 410»), и текст этого пункта берётся отсюда.
RULE_SET_LIBRARY: dict[str, "RuleSet"] = {}


def document_id(text: str) -> str:
    """Обозначение документа без разделителей и регистра: «O GX-STD 410» → «OGXSTD410».

    Из имени папки берутся ведущие части до первого слова в нижнем регистре:
    «OGX-STD-410_multi_equipment_…» → «OGXSTD410», «XYZ_STD_100_normalization_rules» → «XYZSTD100».
    """
    parts = []
    for token in re.split(r"_", str(text)):
        if token and token == token.lower() and not token.isdigit():
            break
        parts.append(token)
    return re.sub(r"[^A-Z0-9]", "", "".join(parts or [str(text)]).upper())


def load_all_rule_sets(settings: Settings, names: set[str] | None = None) -> list[RuleSet]:
    """Находит все <doc_dir>/rule_groups.json и собирает RuleSet на каждый документ.

    names — загрузить только эти документы (подгрузка только что извлечённого документа).
    """
    rule_sets: list[RuleSet] = []

    for path in sorted(settings.rule_groups_dir.glob("*/rule_groups.json")):
        name = path.parent.name
        if names is not None and name not in names:
            continue
        groups: list[dict] = json.loads(path.read_text(encoding="utf-8"))
        global_rules, entries = _split_groups(groups)
        if not entries:
            continue
        parent_to_indices: dict[str, list[int]] = {}
        for i, e in enumerate(entries):
            parent_to_indices.setdefault(e.parent or e.attribute, []).append(i)
        store = load_vector_store(settings, entries, name)
        corpus = [_tokenize(_search_text(e)) for e in entries]
        bm25 = BM25Okapi(corpus)
        deterministic = _load_deterministic(path.parent)
        phrase_map = _load_phrase_rules(path.parent)
        phrase_attributes = _load_phrase_attributes(path.parent)
        tables = load_tables(path.parent)
        roles = _load_table_roles(path.parent)
        _link_cited_lists(roles, groups, tables)
        synonyms, owners, universal = roles.synonyms, roles.synonym_owners, roles.universal
        if synonyms:
            logger.info("Документ %r: перечней синонимов %d, атрибутов с ними %d.",
                        name, len(synonyms), len(owners))
        deterministic_order = _load_deterministic_order(path.parent)
        deterministic_attributes = _load_deterministic_attributes(path.parent)
        closed_registers = _build_closed_registers(phrase_map, deterministic,
                                                   set(deterministic_order),
                                                   deterministic_attributes)
        attribute_to_register = _attribute_to_register_from_origins(
            closed_registers, phrase_attributes, deterministic_attributes)
        if not attribute_to_register:
            attribute_to_register = _derive_attribute_to_register(groups, closed_registers)
        # реестры кодов, которые модель нашла в тексте групп при загрузке документа
        # (table_roles.json); раньше их выискивал в строках правил разбор регулярками
        closed_registers.update(roles.registers)
        for attribute, register_ids in roles.register_owners.items():
            for register_id in register_ids:
                if register_id not in attribute_to_register.setdefault(attribute, []):
                    attribute_to_register[attribute].append(register_id)
        if phrase_map and not closed_registers:
            logger.warning(
                "Документ %r: есть фразовые пары (phrase_rules.json), но ни одна не отнесена "
                "к закрытому регистру — pending-registration не сработает.", name)
        number_first = _determine_number_first(deterministic_order)
        excluded_attributes = _load_excluded_attributes(path.parent)
        render_templates = _load_render_templates(path.parent)
        for attribute, template in _render_templates_from_rules(groups).items():
            render_templates.setdefault(attribute, template)
        for attribute, template in _pattern_templates_from_rules(groups).items():
            render_templates.setdefault(attribute, template)
        varying = _decimals_vary_by_class(groups)
        for attribute in [a for a in render_templates if a.strip().lower() in varying]:
            if render_templates.pop(attribute, None):
                logger.info("Документ %r: шаблон %r снят — знаки зависят от класса оборудования",
                            name, attribute)
        global_card, cards, card_forms = _load_rule_cards(path.parent)
        examples = [{**e, "group": name} for name, card in cards.items()
                    for e in card.get("examples") or []]
        examples += [{**e, "group": ""} for e in global_card.get("examples") or []]
        # форма значения группы — по её примерам из карточки, а не по стрелкам в тексте
        for entry in entries:
            card_examples = (cards.get(entry.attribute) or {}).get("examples") or []
            entry.shapes = {sh for e in card_examples
                            for sh in (value_shape(e["from"]), value_shape(e["to"])) if sh}
        clause_index = {**_build_clause_index(global_rules, entries),
                        **_clause_index_from_pages(path.parent)}
        group_subjects = _load_subjects(path.parent)
        distinct_subjects = {s for s in group_subjects.values() if s}
        if len(distinct_subjects) > 1:
            logger.info("Документ %r описывает несколько предметов: %s",
                        name, sorted(distinct_subjects))
        rule_sets.append(RuleSet(name=name, global_rules=global_rules, store=store, bm25=bm25, groups=entries, deterministic=deterministic, phrase_map=phrase_map, closed_registers=closed_registers, attribute_to_register=attribute_to_register, number_first=number_first, deterministic_order=deterministic_order, deterministic_attributes=deterministic_attributes, excluded_attributes=excluded_attributes, render_templates=render_templates, parent_to_indices=parent_to_indices, clause_index=clause_index, group_subjects=group_subjects, tables=tables, synonyms=synonyms, synonym_owners=owners, universal_synonyms=universal, class_blocks=_class_blocks(groups), global_card=global_card, cards=cards, card_forms=card_forms, examples=examples,
                                 opening_text=_opening_text(path.parent)))
        RULE_SET_LIBRARY[document_id(name)] = rule_sets[-1]

    return rule_sets


OPENING_CHARS = 2500


def _opening_text(doc_dir: Path) -> str:
    """Начало документа — текст блоков первых страниц по порядку, без таблиц.

    Область действия документа стоит в его начале, но не всегда правилом: у служебной записки
    это строка темы («Subject: how we want wellhead, tree and BOP data entered»), у стандарта —
    заголовок и пункт «Scope». Группировка шапку письма в правила не берёт, и вопрос «про это ли
    изделие документ» видел одно «wellhead equipment» — устьевое оборудование, ПВО и фонтанная
    арматура получали пустые ячейки.
    """
    path = doc_dir / "data_after_check.json"
    if not path.exists():
        return ""
    try:
        pages = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return ""
    parts: list[str] = []
    size = 0
    for page in pages:
        content = parse_page_content(page.get("content")) if isinstance(page, dict) else None
        for block in (content or {}).get("blocks", []):
            text = str(block.get("text") or "").strip()
            if block.get("type") == "table" or not text:
                continue
            parts.append(text)
            size += len(text)
            if size >= OPENING_CHARS:
                return "\n".join(parts)[:OPENING_CHARS]
    return "\n".join(parts)


def _load_deterministic(doc_dir: Path) -> dict[str, str]:
    """Загружает детерминированные правила Keyword → Root из папки документа."""
    det_path = doc_dir / "deterministic_rules.json"
    if det_path.exists():
        return json.loads(det_path.read_text(encoding="utf-8"))
    return {}


def _load_phrase_rules(doc_dir: Path) -> dict[str, str]:
    """Загружает фразовые соответствия «фраза → код» (закрытые списки) из папки документа."""
    phrase_path = doc_dir / "phrase_rules.json"
    if phrase_path.exists():
        return json.loads(phrase_path.read_text(encoding="utf-8"))
    return {}


def _load_phrase_attributes(doc_dir: Path) -> dict[str, str]:
    """Загружает происхождение фраз: phrase → attribute (из phrase_attributes.json)."""
    path = doc_dir / "phrase_attributes.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _load_excluded_attributes(doc_dir: Path) -> list[str]:
    """Загружает атрибуты, исключённые из реформатирования (reproduced exactly / excluded)."""
    excl_path = doc_dir / "excluded_attributes.json"
    if excl_path.exists():
        return json.loads(excl_path.read_text(encoding="utf-8"))
    return []


def _load_deterministic_order(doc_dir: Path) -> dict[str, bool]:
    """Загружает порядок конкатенации keyword→root (per-keyword), если файл есть."""
    order_path = doc_dir / "deterministic_order.json"
    if order_path.exists():
        return json.loads(order_path.read_text(encoding="utf-8"))
    return {}


def _load_deterministic_attributes(doc_dir: Path) -> dict[str, str]:
    """Загружает атрибут-владельца для каждого keyword (keyword → attribute)."""
    attrs_path = doc_dir / "deterministic_attributes.json"
    if attrs_path.exists():
        return json.loads(attrs_path.read_text(encoding="utf-8"))
    return {}


def _load_subjects(doc_dir: Path) -> dict[str, str]:
    """Загружает разметку «группа правил → предмет» (subjects.json), если она есть.

    Нужна, когда один документ описывает несколько разных изделий: тогда правила с одинаковым
    названием атрибута относятся к разным вещам и смешивать их нельзя.
    """
    path = doc_dir / "subjects.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _load_render_templates(doc_dir: Path) -> dict[str, dict]:
    """Загружает шаблоны рендера (decimals + suffix) для атрибутов с жёстким форматом."""
    render_path = doc_dir / "render_templates.json"
    if render_path.exists():
        return json.loads(render_path.read_text(encoding="utf-8"))
    return {}


def _extract_code_prefix(code: str) -> str | None:
    """Буквенная часть кода = префикс регистра. Регистр букв и разделители не важны."""
    m = re.match(r"^([A-Za-z]+)", code)
    return m.group(1) if m else None


def _is_code_like(code: str) -> bool:
    """Код ли это (короткий буквенно-цифровой идентификатор с разделителем/цифрой),
    а не длинный root (CLASSATUM, DIAMOSA, KITARIS)."""
    return bool(re.search(r"[-_.\d]", code)) or len(code) <= 4


def _is_indexed_code(code: str) -> bool:
    """Код указывает на конкретный пункт реестра (несёт номер/разделитель), а не является
    чистой буквенной аббревиатурой keyword'а, к которой число добавляется из значения."""
    return bool(re.search(r"[-_./\d]", code))


def _is_register_pair(phrase: str, code: str, keyword_pairs: set[str] = frozenset()) -> bool:
    """Пара «фраза → код закрытого реестра», а не «keyword → invented root».

    keyword_pairs — ключи, для которых извлечение вывело порядок конкатенации: наличие порядка
    и означает, что это keyword-number правило («Pressure class» + «600» → «CL600»), а не пункт
    реестра. Без этого признака различаем по форме: многословная фраза (keyword — одно слово)
    ИЛИ код-индекс с цифрой/разделителем (root — чистые буквы, число приходит из значения).
    """
    if phrase.strip() in keyword_pairs:
        return False
    return _is_code_like(code) and (" " in phrase.strip() or _is_indexed_code(code))


@dataclass
class TableRoles:
    synonyms: dict[str, dict[str, str]] = field(default_factory=dict)  # перечень → форма → запись
    synonym_owners: dict[str, list[str]] = field(default_factory=dict)  # атрибут → перечни
    universal: list[str] = field(default_factory=list)                  # перечни для любого атрибута
    registers: dict[str, dict[str, str]] = field(default_factory=dict)  # реестр → наименование → код
    register_owners: dict[str, list[str]] = field(default_factory=dict)


def _load_table_roles(doc_dir: Path) -> TableRoles:
    """Перечни соответствий значений из table_roles.json (graphs/structured_documents/table_roles.py).

    Роль каждого перечня (реестр кодов, синонимы, пересчёт, примеры, прочее) определила модель
    при загрузке документа, а пары проверены по тексту группы. Код здесь не решает, какая
    колонка — код, а какая — описание: раньше это угадывалось по виду ячейки, и таблица
    коэффициентов пересчёта становилась реестром. Перечень общих положений действует, только
    если модель отметила его как применимый к любому атрибуту; реестры общих положений не берутся.
    Форма, которая в одном перечне ведёт к двум разным записям, отбрасывается: решает модель.
    """
    roles = TableRoles()
    path = doc_dir / "table_roles.json"
    if not path.exists():
        return roles
    try:
        lists = json.loads(path.read_text(encoding="utf-8")).get("lists") or []
    except (json.JSONDecodeError, OSError, AttributeError):
        return roles
    for item in lists:
        role, pairs = item.get("role"), item.get("pairs") or []
        if role not in ("register", "synonyms") or not pairs:
            continue
        # владелец — атрибут, чьи значения перечислены (его называет модель); группа, в тексте
        # которой перечень оказался, — не всегда он: «Detection principle» лежал в группе «Gas»
        group = str(item.get("attribute") or item.get("group") or "").strip()
        is_global = bool(item.get("global"))
        if is_global and (role == "register" or not item.get("any_attribute")):
            continue
        # имя перечня — его заголовок; оно же идёт в колонку «применено»
        title = str(item.get("title") or "").strip()
        base = f"{title} ({group})" if title and group and group.lower() not in title.lower() else (title or group)
        list_id, n = base, 1
        while list_id in roles.synonyms or list_id in roles.registers:
            n += 1
            list_id = f"{base} #{n}"
        if role == "register":
            # пара, чья фраза называет сам атрибут («Class» → корень в группе «class»), — корень
            # обозначения, к которому приписывается номер, а не пункт реестра
            entries = {p["form"]: p["value"] for p in pairs
                       if not _names_its_attribute(p["form"], group)}
            # записанный код — тоже пункт реестра: «Class 150» при паре «150 → Class 150»
            for code in list(entries.values()):
                entries.setdefault(code, code)
            if not entries:
                continue
            roles.registers[list_id] = entries
            roles.register_owners.setdefault(group.lower(), []).append(list_id)
            continue
        mapping: dict[str, str] = {}
        clashing: set[str] = set()
        for pair in pairs:
            form, value = pair["form"], pair["value"]
            if len(form) > MAX_FORM_CHARS or len(value) > MAX_FORM_CHARS:
                continue  # описание, а не форма значения
            for key in (form_key(form), form_key(value)):
                if mapping.setdefault(key, value) != value:
                    clashing.add(key)
        mapping = {k: v for k, v in mapping.items() if k not in clashing}
        if not mapping:
            continue
        roles.synonyms[list_id] = mapping
        if item.get("any_attribute"):
            roles.universal.append(list_id)
        else:
            roles.synonym_owners.setdefault(group.lower(), []).append(list_id)
    return roles


def _link_cited_lists(roles: TableRoles, groups: list[dict], tables: dict[str, DocTable]) -> None:
    """Перечень с номером таблицы в заголовке действует и для атрибутов, которые на неё ссылаются.

    Таблица синонимов часто стоит в своём разделе, а атрибут лишь ссылается на неё: правилом
    своей группы («see Table 12.41») или строкой таблицы-указателя атрибутов («Attribute: 'Roof
    type'; Written form: 'Canonical value of Table 12.41'»). Модель видит перечень в группе, где
    он стоит; кто ещё на него ссылается, видно по номеру таблицы.
    """
    by_number: dict[str, list[str]] = {}
    for list_id in roles.synonyms:
        for number in table_references(list_id):
            by_number.setdefault(number, []).append(list_id)
    if not by_number:
        return

    def add(attribute: str, numbers: list[str]) -> None:
        key = str(attribute).strip().lower()
        for list_id in (l for n in numbers for l in by_number.get(n, [])):
            if key and list_id not in roles.universal \
                    and list_id not in roles.synonym_owners.setdefault(key, []):
                roles.synonym_owners[key].append(list_id)

    for group in groups:
        if group.get("type") != "global":
            add(group.get("attribute", ""), table_references("\n".join(map(str, group.get("rules", [])))))
    for table in tables.values():
        attribute_column = next((c for c in table.columns if c.strip().lower() == "attribute"), None)
        for row in table.rows if attribute_column else []:
            if attribute_column in row:
                add(row[attribute_column], [n for v in row.values() for n in table_references(v)])
            else:
                # имя атрибута распознано ключом строки: {'Sensor type': 'Canonical value of Table 12.46'}
                for key, cell in row.items():
                    if key not in table.columns:
                        add(key, table_references(cell))


def _names_its_attribute(phrase: str, owner: str) -> bool:
    """Фраза называет сам атрибут, а не его значение («Pump size» из группы «pump size»).

    Пункт закрытого реестра описывает ЗНАЧЕНИЕ («Ring-type joint» в группе «end connection»),
    а keyword→root — имя атрибута, к которому приписывается номер. Если извлечение не выдало
    deterministic_order.json, отличить их можно только так — и без этого «Pump size 4»
    сводилось к коду «PS» вместо «PS4».
    """
    p_words = re.findall(r"\w+", str(phrase).lower())
    o_words = re.findall(r"\w+", str(owner).lower())
    return bool(p_words) and bool(o_words) and (p_words == o_words or set(p_words) <= set(o_words))


def _build_closed_registers(phrase_map: dict, deterministic: dict,
                            keyword_pairs: set[str] = frozenset(),
                            owners: dict[str, str] | None = None) -> dict[str, dict[str, str]]:
    """Группирует фразы закрытых регистров по префиксу кода.

    phrase_map — все записи «фраза → код» (это коды закрытого регистра).
    deterministic — может содержать и keyword→root, поэтому фильтруется.
    """
    registers: dict[str, dict[str, str]] = {}
    for phrase, code in phrase_map.items():
        prefix = _extract_code_prefix(code)
        if prefix:
            registers.setdefault(prefix, {})[phrase] = code
    for phrase, code in deterministic.items():
        if _names_its_attribute(phrase, (owners or {}).get(phrase, "")):
            continue
        if _is_register_pair(phrase, code, keyword_pairs):
            prefix = _extract_code_prefix(code)
            if prefix:
                registers.setdefault(prefix, {})[phrase] = code
    return registers


def _attribute_to_register_from_origins(
    closed_registers: dict[str, dict[str, str]], *origins: dict[str, str]
) -> dict[str, list[str]]:
    """attribute → [префиксы] из происхождения фраз, без сканирования прозы.

    Источники происхождения (phrase_attributes.json, deterministic_attributes.json) —
    любые словари «фраза → атрибут, из группы которого она извлечена». Пара попадает в
    маппинг только если сама фраза действительно оказалась в закрытом регистре.
    """
    mapping: dict[str, list[str]] = {}
    for prefix, phrases in closed_registers.items():
        for phrase in phrases:
            attr = next((o[phrase] for o in origins if o.get(phrase)), None)
            if attr and prefix not in mapping.setdefault(attr.strip().lower(), []):
                mapping[attr.strip().lower()].append(prefix)
    return mapping


def _derive_attribute_to_register(groups: list[dict], closed_registers: dict[str, dict[str, str]]) -> dict[str, list[str]]:
    """Fallback: attribute → [префиксы] сканированием текста правил (Code: JT-01 / → "ECC-101" / …)."""
    prefixes = list(closed_registers.keys())
    if not prefixes:
        return {}
    alt = "|".join(re.escape(p) for p in sorted(prefixes, key=len, reverse=True))
    pat = re.compile(
        r"(?:Code(?:\s+Block)?\s*:\s*|Assigned\s+Codes\s*:\s*|→\s*)"
        r"[“”‘’'\"]?"
        r"(" + alt + r")-?\d+\b"
    )
    mapping: dict[str, list[str]] = {}
    for g in groups:
        if g.get("type") == "global":
            continue
        for r in g.get("rules", []):
            m = pat.search(r)
            if m:
                mapping.setdefault(g["attribute"].strip().lower(), []).append(m.group(1))
                break
    return mapping


_UNIT_CELL_RE = re.compile(r"\bunits?\s*[:=]\s*[\'\"“]?([^;,\'\"”\s]{1,10})", re.I)
_DECIMALS_CELL_RE = re.compile(r"\bdecimals?(?:\s+places)?\s*[:=]\s*[\'\"“]?(\d)", re.I)


def _render_templates_from_rules(groups: list[dict]) -> dict[str, dict]:
    """Шаблоны рендера из строк-таблиц «величина — единица — знаки после запятой».

    Берём строку, только если её подпись совпадает с именем группы: в таблице, расписанной по
    классам оборудования («Valves (all types) — in. — 3»), подпись к атрибуту не относится, и
    навязывать её как шаблон нельзя.
    """
    templates: dict[str, dict] = {}
    for group in groups:
        attribute = (group.get("attribute") or "").strip()
        if group.get("type") == "global" or not attribute:
            continue
        words = set(re.findall(r"\w+", attribute.lower()))
        for rule in group.get("rules", []):
            for line in str(rule).splitlines():
                unit = _UNIT_CELL_RE.search(line)
                decimals = _DECIMALS_CELL_RE.search(line)
                if not unit or not decimals:
                    continue
                label = re.split(r"[—\-|:]", line, 1)[0]
                label_words = set(re.findall(r"\w+", label.lower()))
                if not label_words or not (label_words & words):
                    continue
                templates.setdefault(attribute.lower(), {
                    "decimals": int(decimals.group(1)),
                    "suffix": " " + unit.group(1),
                    "aliases": [a for a in group.get("aliases", []) if a],
                })
    return templates


# Форма значения, заданная шаблоном: «<n.n> kW», «<n> Pa» — вся строка, без оговорок после.
_PATTERN_FORM_RE = re.compile(
    r"(?:^|[\s:'\"“])<n(?:\.(?P<dec>n+))?>(?P<space>\s?)(?P<unit>[^\s<>'\"“”()\[\];,]{1,8})\s*['\"”]?\s*\.?\s*$")
# Ячейка «Decimal places: '1'» в строке таблицы, расписанной по классам оборудования.
_DECIMALS_VALUE_RE = re.compile(r"\bdecimal(?:s| places)?[^:;]*:\s*['\"“]?(\d)\b", re.I)


def _pattern_templates_from_rules(groups: list[dict]) -> dict[str, dict]:
    """Шаблоны рендера из словаря форм («Rated power | <n.n> kW»).

    Иной документ задаёт число знаков только так — в приложении, а в самом пункте пишет «в форме,
    данной в Appendix A». Без шаблона «9.0 kW» и «9 kW» сводились друг к другу как попало.
    Строка с оговоркой после формы («<n.n> m3/h (decimals per Table 1)») шаблоном не считается.
    """
    templates: dict[str, dict] = {}
    for group in groups:
        attribute = (group.get("attribute") or "").strip()
        if group.get("type") == "global" or not attribute:
            continue
        for rule in group.get("rules", []):
            for line in str(rule).splitlines():
                m = _PATTERN_FORM_RE.search(line)
                if m:
                    templates.setdefault(attribute.lower(), {
                        "decimals": len(m.group("dec") or ""),
                        "suffix": m.group("space") + m.group("unit"),
                        "aliases": [a for a in group.get("aliases", []) if a],
                    })
    return templates


def _decimals_vary_by_class(groups: list[dict]) -> set[str]:
    """Группы, где число знаков задано таблицей по классам оборудования и различается.

    Шаблон из одного примера («45 m3/h» → «45.0 m3/h», насос) у такой группы верен лишь для
    одного класса: у чиллера того же документа знаков ноль. Подставлять его всем нельзя.
    """
    varying: set[str] = set()
    for group in groups:
        values = {m.group(1) for rule in group.get("rules", [])
                  for m in _DECIMALS_VALUE_RE.finditer(str(rule))}
        if len(values) > 1:
            varying.add((group.get("attribute") or "").strip().lower())
    return varying


def _determine_number_first(deterministic_order: dict[str, bool]) -> bool:
    """Порядок по умолчанию для keyword, которому извлечение порядок не вывело.

    Берётся большинством по тем keyword, где порядок известен: внутри документа конвенция
    обычно одна. Раньше здесь искалась фраза «numeral first» в тексте правил — это была
    привязка к формулировке конкретного документа, и для документа с другой формулировкой
    она молча давала неверный порядок.
    """
    if not deterministic_order:
        return False
    return sum(bool(v) for v in deterministic_order.values()) * 2 > len(deterministic_order)


def hybrid_search(rule_set: RuleSet, query: str, top_k: int = 4,
                  k_semantic: int = 8, k_bm25: int = 8,
                  min_score: float = 0.02) -> list[str]:
    """Гибридный поиск: semantic + BM25, слияние через RRF. Возвращает тела правил.

    min_score — порог по RRF-score: группы со скором ниже отбрасываются, чтобы не
    тянуть нерелевантный/противоречащий контекст в промпт модели.
    """
    # semantic
    sem_docs = rule_set.store.similarity_search(query, k=k_semantic)
    sem_ranks = [(d.metadata.get("index", -1), rank) for rank, d in enumerate(sem_docs, start=1)]

    # bm25
    scores = rule_set.bm25.get_scores(_tokenize(query))
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    bm25_ranks = [(i, rank) for rank, i in enumerate(order[:k_bm25], start=1)]

    # RRF
    rrf: dict[int, float] = {}
    for idx, rank in sem_ranks + bm25_ranks:
        if 0 <= idx < len(rule_set.groups):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (60 + rank)

    top = [(i, s) for i, s in sorted(rrf.items(), key=lambda kv: -kv[1])[:top_k] if s >= min_score]
    return [rule_set.groups[i].body for i, _ in top]


def add_rule_set(app_config: AppConfig, name: str) -> RuleSet | None:
    """Подгружает документ в работающий сервис (после фонового извлечения); заменяет прежний с тем же именем."""
    loaded = load_all_rule_sets(app_config.settings, {name})
    if not loaded:
        return None
    app_config.rule_sets[:] = [rs for rs in app_config.rule_sets if rs.name != name] + loaded
    return loaded[0]


def build_app_config(settings: Settings | None = None) -> AppConfig:
    settings = settings or Settings()

    assistant = ChatOpenAI(
        model=settings.chat_model,
        temperature=settings.temperature,
        reasoning_effort=settings.reasoning_effort,
        # без таймаута SDK ждёт повисший запрос по 600 с на попытку — прогон стоял полчаса
        timeout=120,
        max_retries=2,
    )
    model = assistant.bind_tools(array_of_tools)
    spec_model = ChatOpenAI(
        model=settings.chat_model,
        temperature=settings.temperature,
        reasoning_effort=settings.spec_reasoning_effort,
        # с рассуждением ответ дольше: запас к таймауту разговорной модели
        timeout=180,
        max_retries=2,
    )

    rule_sets = load_all_rule_sets(settings)

    return AppConfig(
        settings=settings,
        model=model,
        rule_sets=rule_sets,
        assistant=assistant,
        spec_model=spec_model,
    )
