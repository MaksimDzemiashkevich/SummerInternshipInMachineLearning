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
    """
    global_rules: list[str] = []
    entries: list[GroupEntry] = []

    for group in groups:
        header = group.get("attribute", "").strip()
        body = "\n".join(rule.strip() for rule in group.get("rules", []))

        if group.get("type") == "global":
            global_rules.append(body)
        else:
            if _is_non_rule(header):
                continue
            aliases = [a.strip() for a in group.get("aliases", []) if a and a.strip()]
            entries.append(GroupEntry(attribute=header, aliases=aliases, body=body, parent=group.get("parent", "") or header))

    return "\n".join(global_rules), entries


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
        # таблицы «описание → код», оставшиеся строками правил, тоже образуют закрытый реестр
        table_pairs, table_attributes = _extract_table_pairs(groups)
        if table_pairs:
            logger.info("Документ %r: из текста правил добавлено %d пар закрытого реестра.",
                        name, len(table_pairs))
        phrase_map = {**table_pairs, **phrase_map}
        phrase_attributes = {**table_attributes, **phrase_attributes}
        deterministic_order = _load_deterministic_order(path.parent)
        closed_registers = _build_closed_registers(phrase_map, deterministic,
                                                   set(deterministic_order))
        deterministic_attributes = _load_deterministic_attributes(path.parent)
        attribute_to_register = _attribute_to_register_from_origins(
            closed_registers, phrase_attributes, deterministic_attributes)
        if not attribute_to_register:
            attribute_to_register = _derive_attribute_to_register(groups, closed_registers)
        if phrase_map and not closed_registers:
            logger.warning(
                "Документ %r: есть фразовые пары (phrase_rules.json), но ни одна не отнесена "
                "к закрытому регистру — pending-registration не сработает.", name)
        number_first = _determine_number_first(global_rules)
        excluded_attributes = _load_excluded_attributes(path.parent)
        render_templates = _load_render_templates(path.parent)
        clause_index = _build_clause_index(global_rules, entries)
        group_subjects = _load_subjects(path.parent)
        distinct_subjects = {s for s in group_subjects.values() if s}
        if len(distinct_subjects) > 1:
            logger.info("Документ %r описывает несколько предметов: %s",
                        name, sorted(distinct_subjects))
        rule_sets.append(RuleSet(name=name, global_rules=global_rules, store=store, bm25=bm25, groups=entries, deterministic=deterministic, phrase_map=phrase_map, closed_registers=closed_registers, attribute_to_register=attribute_to_register, number_first=number_first, deterministic_order=deterministic_order, deterministic_attributes=deterministic_attributes, excluded_attributes=excluded_attributes, render_templates=render_templates, parent_to_indices=parent_to_indices, clause_index=clause_index, group_subjects=group_subjects))

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
    return bool(text) and " " not in text and (_is_indexed_code(text) or len(text) <= 6)


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


def _extract_table_pairs(groups: list[dict]) -> tuple[dict[str, str], dict[str, str]]:
    """Достаёт из тела local-групп строки таблиц «описание → код» вместе с их атрибутом.

    Документ может не выносить такую таблицу в отдельный файл на этапе извлечения, оставив
    её строками правил. Без этого закрытый реестр для атрибута не строится вовсе, и код
    подбирает LLM по тексту — то есть может выдумать отсутствующий в таблице код.
    """
    pairs: dict[str, str] = {}
    origins: dict[str, str] = {}
    for group in groups:
        if group.get("type") == "global":
            continue
        attribute = (group.get("attribute") or "").strip()
        for rule in group.get("rules", []):
            for line in str(rule).splitlines():
                cells = [c.strip() for c in _TABLE_CELL_RE.findall(line)]
                if len(cells) < 2:
                    continue
                oriented = _orient_table_row(cells)
                if not oriented:
                    continue
                phrase, code = oriented
                if phrase and code and _is_register_pair(phrase, code):
                    pairs[phrase] = code
                    origins[phrase] = attribute
    return pairs, origins


def _build_closed_registers(phrase_map: dict, deterministic: dict,
                            keyword_pairs: set[str] = frozenset()) -> dict[str, dict[str, str]]:
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


def _determine_number_first(global_rules: str) -> bool:
    """Определяет порядок keyword→root по тексту глобальных правил:
    number-first («numeral first» → 300CLASSATUM) или root-first (CL300)."""
    return bool(re.search(r"numeral\s+first", global_rules.lower()))


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
    )
    model = assistant.bind_tools(array_of_tools)

    rule_sets = load_all_rule_sets(settings)

    return AppConfig(
        settings=settings,
        model=model,
        rule_sets=rule_sets,
        assistant=assistant,
    )
