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

from .tables import DocTable, load_tables, synonym_map, synonym_owners
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


@dataclass
class GroupEntry:
    """Одна группа правил: имя атрибута, алиасы, текст правил и родительская тема."""
    attribute: str
    aliases: list[str]
    body: str
    parent: str = ""
    shapes: set[str] = field(default_factory=set)


_SHAPE_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
_SHAPE_EXAMPLE_RE = re.compile(
    r"[\"“']([^\"“”']{1,80})[\"”']\s*(?:->|→|becomes)\s*[\"“']([^\"“”']{1,80})[\"”']")


def value_shape(text: str) -> str:
    """Форма числового значения: числа → «9», без пробелов и регистра.

    «600 x 600mm» и «500 x 1200mm» — одна форма «9x9mm», «2.5 bar» и «1.5 bar» — «9bar».
    У значения без цифр формы нет: по словам совпадает слишком многое.
    """
    text = str(text).strip().lower()
    if not re.search(r"\d", text):
        return ""
    return re.sub(r"\s+", "", _SHAPE_NUMBER_RE.sub("9", text))


def _example_shapes(body: str) -> set[str]:
    """Формы обеих сторон примеров группы («"2.5 bar" → "250 kPa"» → {"9bar", "9kpa"})."""
    shapes = set()
    for left, right in _SHAPE_EXAMPLE_RE.findall(body):
        shapes.update(s for s in (value_shape(left), value_shape(right)) if s)
    return shapes


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


@dataclass
class AppConfig:
    settings: Settings
    model: ChatOpenAI
    rule_sets: list[RuleSet]
    # Та же модель без bind_tools: разговорной части тулзы конверсии единиц не нужны.
    assistant: ChatOpenAI | None = None


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
                                      parent=group.get("parent", "") or header,
                                      shapes=_example_shapes(body)))

    return "\n".join(global_rules), entries


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


def load_all_rule_sets(settings: Settings) -> list[RuleSet]:
    """Находит все <doc_dir>/rule_groups.json и собирает RuleSet на каждый документ."""
    rule_sets: list[RuleSet] = []

    for path in sorted(settings.rule_groups_dir.glob("*/rule_groups.json")):
        name = path.parent.name
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
        synonyms = {tid: m for tid, table in tables.items() if (m := synonym_map(table))}
        owners, universal = synonym_owners(groups, tables, synonyms)
        if synonyms:
            logger.info("Документ %r: таблиц синонимов %d, атрибутов с ними %d.",
                        name, len(synonyms), len(owners))
        # таблицы «описание → код», оставшиеся строками правил, тоже образуют закрытый реестр;
        # строки таблиц синонимов туда не идут — там направление обратное (много форм → одна)
        table_pairs, table_attributes = _extract_table_pairs(
            groups, deterministic, [tables[t].title for t in synonyms],
            [{c.strip().lower() for c in tables[t].columns} for t in synonyms])
        if table_pairs:
            logger.info("Документ %r: из текста правил добавлено %d пар закрытого реестра.",
                        name, len(table_pairs))
        phrase_map = {**table_pairs, **phrase_map}
        phrase_attributes = {**table_attributes, **phrase_attributes}
        deterministic_order = _load_deterministic_order(path.parent)
        deterministic_attributes = _load_deterministic_attributes(path.parent)
        closed_registers = _build_closed_registers(phrase_map, deterministic,
                                                   set(deterministic_order),
                                                   deterministic_attributes)
        attribute_to_register = _attribute_to_register_from_origins(
            closed_registers, phrase_attributes, deterministic_attributes)
        if not attribute_to_register:
            attribute_to_register = _derive_attribute_to_register(groups, closed_registers)
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
        clause_index = {**_build_clause_index(global_rules, entries),
                        **_clause_index_from_pages(path.parent)}
        group_subjects = _load_subjects(path.parent)
        distinct_subjects = {s for s in group_subjects.values() if s}
        if len(distinct_subjects) > 1:
            logger.info("Документ %r описывает несколько предметов: %s",
                        name, sorted(distinct_subjects))
        rule_sets.append(RuleSet(name=name, global_rules=global_rules, store=store, bm25=bm25, groups=entries, deterministic=deterministic, phrase_map=phrase_map, closed_registers=closed_registers, attribute_to_register=attribute_to_register, number_first=number_first, deterministic_order=deterministic_order, deterministic_attributes=deterministic_attributes, excluded_attributes=excluded_attributes, render_templates=render_templates, parent_to_indices=parent_to_indices, clause_index=clause_index, group_subjects=group_subjects, tables=tables, synonyms=synonyms, synonym_owners=owners, universal_synonyms=universal, class_blocks=_class_blocks(groups)))
        RULE_SET_LIBRARY[document_id(name)] = rule_sets[-1]

    return rule_sets


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


# Ячейка строки-таблицы: «<подпись колонки>: '<значение>'». Колонок бывает две и больше.
_TABLE_CELL_RE = re.compile(r"[\w /&()-]+?:\s*['\"“”](.+?)['\"“”]\s*(?:;|$)")


def _looks_like_code(text: str) -> bool:
    """Похоже ли значение ячейки на идентификатор, а не на описание."""
    text = text.strip()
    # «Close-coupled» — слово через дефис, а не код: у кодов нет строчных слов
    return (bool(text) and " " not in text and not re.search(r"[a-z]{3,}", text)
            and (_is_indexed_code(text) or len(text) <= 6))


def _orient_table_row(cells: list[str]) -> tuple[str, str] | None:
    """Возвращает (описание, код) из ячеек строки таблицы, сколько бы колонок в ней ни было.

    Порядок колонок в документе произвольный: «Joint Type: 'X'; Abbreviation: 'Y'» и
    «Code: 'Y'; Description: 'X'» — одно и то же, поэтому роль колонки определяется по виду
    значения, а не по позиции. Если описания нет или кодов сразу несколько («Numeric code:
    'ECC-101'; Description: '...'; Abbreviation: 'RF'»), решить, какой из них требуется,
    по одной строке нельзя — такую таблицу не трогаем, пусть её читает модель.
    """
    codes = [c for c in cells if _looks_like_code(c)]
    descriptions = [c for c in cells if not _looks_like_code(c)]
    if len(codes) != 1 or len(descriptions) != 1:
        return None
    return descriptions[0], codes[0]


def _split_table_row(line: str) -> list[str]:
    """Ячейки строки таблицы в любой из двух форм, которыми её оставляет извлечение.

    Первая — «Joint Type: 'X'; Abbreviation: 'Y'», вторая — колонки через вертикальную черту
    («Electric motor, squirrel cage | DRV-EM»). Вторую раньше не разбирали вовсе, и таблица
    кодов такого документа просто не становилась закрытым реестром.
    """
    cells = [c.strip() for c in _TABLE_CELL_RE.findall(line)]
    if len(cells) >= 2:
        return cells
    if "|" in line:
        cells = [c.strip() for c in line.split("|")]
        return cells if 2 <= len(cells) <= 4 and all(c and len(c) <= 60 for c in cells) else []
    # третья форма: «Описание -> КОД». Строку примера пропускаем: там стрелка показывает
    # преобразование значения, а не строку таблицы.
    if re.search(r"->|→", line) and not re.match(r"\s*EXAMPLE", line, re.I):
        cells = [c.strip().strip("\"'“”") for c in re.split(r"->|→", line)]
        return cells if len(cells) == 2 and all(c and len(c) <= 60 for c in cells) else []
    return []


# Подписи ячеек строки таблицы: «<подпись>: '<значение>'».
_CELL_LABEL_RE = re.compile(r"(?:^|;\s*)(?:\[[^\]]*\]\s*)?([\w /&()-]+?):\s*['\"“”]")
# Колонка, которая держит ГОТОВУЮ форму значения («Written form», «Canonical value»,
# «Normalized value»): строка такой таблицы читается от записи к этой форме, и прочесть её как
# «описание → код» значит перевернуть — «SCH STD» стал описанием кода «STD».
_RESULT_LABEL_RE = re.compile(r"\b(?:written form|canonical|normali[sz]ed)\b")


def _extract_table_pairs(groups: list[dict],
                         deterministic: dict[str, str] | None = None,
                         skip_titles: list[str] | None = None,
                         skip_columns: list[set[str]] | None = None
                         ) -> tuple[dict[str, str], dict[str, str]]:
    """Достаёт из тела local-групп строки таблиц «описание → код» вместе с их атрибутом.

    Документ может не выносить такую таблицу в отдельный файл на этапе извлечения, оставив
    её строками правил. Без этого закрытый реестр для атрибута не строится вовсе, и код
    подбирает LLM по тексту — то есть может выдумать отсутствующий в таблице код.
    Пары keyword→root (таблица корней обозначений) сюда не попадают: это не список значений,
    и в закрытом реестре им не место.
    """
    roots = {k.strip().lower() for k in (deterministic or {})}
    pairs: dict[str, str] = {}
    origins: dict[str, str] = {}
    for group in groups:
        if group.get("type") == "global":
            continue
        attribute = (group.get("attribute") or "").strip()
        for rule in group.get("rules", []):
            for line in str(rule).splitlines():
                if any(t and t in line for t in skip_titles or ()):
                    continue
                labels = {l.strip().lower() for l in _CELL_LABEL_RE.findall(line)}
                if any(labels == c for c in skip_columns or ()) or any(
                        _RESULT_LABEL_RE.search(l) for l in labels):
                    continue
                cells = _split_table_row(line)
                if len(cells) < 2:
                    continue
                oriented = _orient_table_row(cells)
                if not oriented:
                    continue
                phrase, code = oriented
                if phrase.strip().lower() in roots:
                    continue  # это корень обозначения, а не пункт списка значений
                if phrase and code and _is_register_pair(phrase, code):
                    pairs[phrase] = code
                    origins[phrase] = attribute
    return pairs, origins


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

    rule_sets = load_all_rule_sets(settings)

    return AppConfig(
        settings=settings,
        model=model,
        rule_sets=rule_sets,
        assistant=assistant,
    )
