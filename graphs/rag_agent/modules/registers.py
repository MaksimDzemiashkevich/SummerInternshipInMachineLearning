"""Данные документа для запасного пути (нормализация графом): реестры кодов, перечни синонимов,
корни обозначений, шаблоны вывода чисел, карточки правил.

Путь по паспортам атрибутов этим не пользуется: всё нужное ему лежит в паспорте атрибута
(attribute_spec.json) и в перечнях table_roles.json, которые исполнитель паспортов читает сам.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from .tables import MAX_FORM_CHARS, DocTable, form_key, table_references

logger = logging.getLogger(__name__)


def read_json(doc_dir: Path, name: str, default):
    """Файл извлечения из папки документа или default, если его нет или он не читается."""
    path = doc_dir / name
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


# ---------------------------------------------------------------- карточки правил
def load_rule_cards(doc_dir: Path) -> tuple[dict, dict[str, dict], dict[str, dict[str, str]]]:
    """(общая карточка, карточки групп, принятые формы групп) из rule_cards.json.

    Для замены формы на запись берутся только пары, чья запись входит в перечень значений самой
    карточки: карточку пишет модель, и пара «Cv → the number only» — описание шага, а не значение.
    """
    data = read_json(doc_dir, "rule_cards.json", {})
    cards = data.get("groups") or {}
    forms: dict[str, dict[str, str]] = {}
    for name, card in cards.items():
        values = {form_key(v): str(v).strip() for v in card.get("values") or [] if str(v).strip()}
        mapping: dict[str, str] = {}
        for pair in card.get("accepted") or []:
            written = values.get(form_key(pair.get("written", "")))
            if written:
                mapping.setdefault(form_key(pair.get("form", "")), written)
        for key, written in values.items():
            mapping.setdefault(key, written)  # сама запись в другом регистре — тоже она
        if mapping:
            forms[name.strip().lower()] = mapping
    return data.get("global") or {}, cards, forms


# ---------------------------------------------------------------- перечни соответствий значений
@dataclass
class TableRoles:
    """Перечни соответствий значений документа, разобранные по ролям (table_roles.json)."""
    synonyms: dict[str, dict[str, str]] = field(default_factory=dict)  # перечень → форма → запись
    synonym_owners: dict[str, list[str]] = field(default_factory=dict)  # атрибут → перечни
    universal: list[str] = field(default_factory=list)                  # перечни для любого атрибута
    registers: dict[str, dict[str, str]] = field(default_factory=dict)  # реестр → наименование → код
    register_owners: dict[str, list[str]] = field(default_factory=dict)


def load_table_roles(doc_dir: Path) -> TableRoles:
    """Реестры кодов и перечни синонимов документа из table_roles.json.

    Роль каждого перечня определила модель при извлечении, пары проверены по тексту группы; код
    здесь не угадывает, какая колонка — код, а какая — описание. Перечень общих положений
    действует, только если модель отметила его применимым к любому атрибуту. Форма, ведущая в
    одном перечне к двум разным записям, отбрасывается: такой случай решает модель.
    """
    roles = TableRoles()
    data = read_json(doc_dir, "table_roles.json", {})
    lists = data.get("lists") or [] if isinstance(data, dict) else []
    for item in lists:
        role, pairs = item.get("role"), item.get("pairs") or []
        if role not in ("register", "synonyms") or not pairs:
            continue
        # владелец — атрибут, чьи значения перечислены; группа, где перечень лежит, — не всегда он
        group = str(item.get("attribute") or item.get("group") or "").strip()
        if item.get("global") and (role == "register" or not item.get("any_attribute")):
            continue
        list_id = _unique_list_id(roles, str(item.get("title") or "").strip(), group)
        if role == "register":
            _add_register(roles, list_id, group, pairs)
        else:
            _add_synonyms(roles, list_id, group, pairs, bool(item.get("any_attribute")))
    return roles


def _unique_list_id(roles: TableRoles, title: str, group: str) -> str:
    """Имя перечня: его заголовок (с атрибутом, если заголовок его не называет), уникальное в документе."""
    base = f"{title} ({group})" if title and group and group.lower() not in title.lower() else (title or group)
    list_id, n = base, 1
    while list_id in roles.synonyms or list_id in roles.registers:
        n += 1
        list_id = f"{base} #{n}"
    return list_id


def _add_register(roles: TableRoles, list_id: str, group: str, pairs: list[dict]) -> None:
    """Реестр «наименование → код». Пара, чья фраза называет сам атрибут, — корень обозначения
    (к нему приписывается номер), а не пункт реестра; записанный код — тоже пункт реестра."""
    entries = {p["form"]: p["value"] for p in pairs if not names_its_attribute(p["form"], group)}
    for code in list(entries.values()):
        entries.setdefault(code, code)
    if entries:
        roles.registers[list_id] = entries
        roles.register_owners.setdefault(group.lower(), []).append(list_id)


def _add_synonyms(roles: TableRoles, list_id: str, group: str, pairs: list[dict], any_attribute: bool) -> None:
    """Перечень «принятая форма → запись»; формы, ведущие к двум записям, и описания вместо форм отбрасываются."""
    mapping: dict[str, str] = {}
    clashing: set[str] = set()
    for pair in pairs:
        form, value = pair["form"], pair["value"]
        if len(form) > MAX_FORM_CHARS or len(value) > MAX_FORM_CHARS:
            continue
        for key in (form_key(form), form_key(value)):
            if mapping.setdefault(key, value) != value:
                clashing.add(key)
    mapping = {k: v for k, v in mapping.items() if k not in clashing}
    if not mapping:
        return
    roles.synonyms[list_id] = mapping
    if any_attribute:
        roles.universal.append(list_id)
    else:
        roles.synonym_owners.setdefault(group.lower(), []).append(list_id)


def link_cited_lists(roles: TableRoles, groups: list[dict], tables: dict[str, DocTable]) -> None:
    """Перечень с номером таблицы в заголовке действует и для атрибутов, которые на неё ссылаются:
    правилом своей группы («see Table 12.41») или строкой таблицы-указателя атрибутов
    («Attribute: 'Roof type'; Written form: 'Canonical value of Table 12.41'»)."""
    by_number: dict[str, list[str]] = {}
    for list_id in roles.synonyms:
        for number in table_references(list_id):
            by_number.setdefault(number, []).append(list_id)
    if not by_number:
        return

    def add(attribute: str, numbers: list[str]) -> None:
        key = str(attribute).strip().lower()
        for list_id in (l for n in numbers for l in by_number.get(n, [])):
            if key and list_id not in roles.universal and list_id not in roles.synonym_owners.setdefault(key, []):
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


# ---------------------------------------------------------------- закрытые реестры и корни обозначений
def names_its_attribute(phrase: str, owner: str) -> bool:
    """Фраза называет сам атрибут, а не его значение («Pump size» из группы «pump size»).

    Пункт закрытого реестра описывает значение («Ring-type joint» в группе «end connection»), а
    keyword→root — имя атрибута, к которому приписывается номер («Pump size 4» → «PS4»).
    """
    p_words = re.findall(r"\w+", str(phrase).lower())
    o_words = re.findall(r"\w+", str(owner).lower())
    return bool(p_words) and bool(o_words) and (p_words == o_words or set(p_words) <= set(o_words))


def _code_prefix(code: str) -> str | None:
    """Буквенная часть кода — префикс реестра (регистр и разделители не важны)."""
    m = re.match(r"^([A-Za-z]+)", code)
    return m.group(1) if m else None


def _is_register_pair(phrase: str, code: str, keyword_pairs: set[str] = frozenset()) -> bool:
    """Пара «фраза → код закрытого реестра», а не «keyword → корень обозначения».

    keyword_pairs — ключи, для которых извлечение вывело порядок «корень + номер»: это правила
    обозначения. Иначе — по форме: код похож на код (короткий, с цифрой или разделителем) и при этом
    фраза многословная или код несёт номер (у корня одни буквы, номер приходит из значения).
    """
    if phrase.strip() in keyword_pairs:
        return False
    code_like = bool(re.search(r"[-_.\d]", code)) or len(code) <= 4
    indexed = bool(re.search(r"[-_./\d]", code))
    return code_like and (" " in phrase.strip() or indexed)


def build_closed_registers(phrase_map: dict, deterministic: dict, keyword_pairs: set[str] = frozenset(),
                           owners: dict[str, str] | None = None) -> dict[str, dict[str, str]]:
    """Закрытые реестры документа: фразы «фраза → код», сгруппированные по префиксу кода.

    phrase_map — все пары закрытых реестров; deterministic смешивает пункты реестров с корнями
    обозначений, поэтому из него берутся только пары реестра.
    """
    registers: dict[str, dict[str, str]] = {}
    for phrase, code in phrase_map.items():
        prefix = _code_prefix(code)
        if prefix:
            registers.setdefault(prefix, {})[phrase] = code
    for phrase, code in deterministic.items():
        if names_its_attribute(phrase, (owners or {}).get(phrase, "")) or not _is_register_pair(phrase, code, keyword_pairs):
            continue
        prefix = _code_prefix(code)
        if prefix:
            registers.setdefault(prefix, {})[phrase] = code
    return registers


def attribute_registers_from_origins(closed_registers: dict[str, dict[str, str]],
                                     *origins: dict[str, str]) -> dict[str, list[str]]:
    """Атрибут → его реестры по происхождению фраз (из группы какого атрибута извлечена фраза)."""
    mapping: dict[str, list[str]] = {}
    for prefix, phrases in closed_registers.items():
        for phrase in phrases:
            attribute = next((o[phrase] for o in origins if o.get(phrase)), None)
            if attribute and prefix not in mapping.setdefault(attribute.strip().lower(), []):
                mapping[attribute.strip().lower()].append(prefix)
    return mapping


def attribute_registers_from_text(groups: list[dict], closed_registers: dict[str, dict[str, str]]) -> dict[str, list[str]]:
    """Атрибут → его реестры по ссылкам на код в тексте правил («Code: JT-01», «→ ECC-101»);
    используется, когда происхождение фраз не сохранено."""
    prefixes = list(closed_registers.keys())
    if not prefixes:
        return {}
    alternatives = "|".join(re.escape(p) for p in sorted(prefixes, key=len, reverse=True))
    pattern = re.compile(r"(?:Code(?:\s+Block)?\s*:\s*|Assigned\s+Codes\s*:\s*|→\s*)[“”‘’'\"]?"
                         r"(" + alternatives + r")-?\d+\b")
    mapping: dict[str, list[str]] = {}
    for group in groups:
        if group.get("type") == "global":
            continue
        for rule in group.get("rules", []):
            m = pattern.search(rule)
            if m:
                mapping.setdefault(group["attribute"].strip().lower(), []).append(m.group(1))
                break
    return mapping


def number_first_default(deterministic_order: dict[str, bool]) -> bool:
    """Порядок «номер + корень» по умолчанию — большинством по корням, чей порядок известен:
    внутри документа конвенция обычно одна."""
    if not deterministic_order:
        return False
    return sum(bool(v) for v in deterministic_order.values()) * 2 > len(deterministic_order)


# ---------------------------------------------------------------- шаблоны вывода чисел
_UNIT_CELL_RE = re.compile(r"\bunits?\s*[:=]\s*[\'\"“]?([^;,\'\"”\s]{1,10})", re.I)
_DECIMALS_CELL_RE = re.compile(r"\bdecimals?(?:\s+places)?\s*[:=]\s*[\'\"“]?(\d)", re.I)
# форма значения, заданная шаблоном: «<n.n> kW», «<n> Pa» — вся строка, без оговорок после
_PATTERN_FORM_RE = re.compile(
    r"(?:^|[\s:'\"“])<n(?:\.(?P<dec>n+))?>(?P<space>\s?)(?P<unit>[^\s<>'\"“”()\[\];,]{1,8})\s*['\"”]?\s*\.?\s*$")
# ячейка «Decimal places: '1'» в строке таблицы, расписанной по классам оборудования
_DECIMALS_VALUE_RE = re.compile(r"\bdecimal(?:s| places)?[^:;]*:\s*['\"“]?(\d)\b", re.I)


def _templates_from_units_table(groups: list[dict]) -> dict[str, dict]:
    """Шаблоны из строк таблицы «величина — единица — знаки», если подпись строки называет группу:
    в таблице, расписанной по классам оборудования, подпись к атрибуту не относится."""
    templates: dict[str, dict] = {}
    for group in groups:
        attribute = (group.get("attribute") or "").strip()
        if group.get("type") == "global" or not attribute:
            continue
        words = set(re.findall(r"\w+", attribute.lower()))
        for line in (l for rule in group.get("rules", []) for l in str(rule).splitlines()):
            unit, decimals = _UNIT_CELL_RE.search(line), _DECIMALS_CELL_RE.search(line)
            label_words = set(re.findall(r"\w+", re.split(r"[—\-|:]", line, 1)[0].lower()))
            if unit and decimals and label_words & words:
                templates.setdefault(attribute.lower(), {
                    "decimals": int(decimals.group(1)), "suffix": " " + unit.group(1),
                    "aliases": [a for a in group.get("aliases", []) if a]})
    return templates


def _templates_from_value_forms(groups: list[dict]) -> dict[str, dict]:
    """Шаблоны из словаря форм («Rated power | <n.n> kW»); строка с оговоркой после формы
    («<n.n> m3/h (decimals per Table 1)») шаблоном не считается."""
    templates: dict[str, dict] = {}
    for group in groups:
        attribute = (group.get("attribute") or "").strip()
        if group.get("type") == "global" or not attribute:
            continue
        for line in (l for rule in group.get("rules", []) for l in str(rule).splitlines()):
            m = _PATTERN_FORM_RE.search(line)
            if m:
                templates.setdefault(attribute.lower(), {
                    "decimals": len(m.group("dec") or ""), "suffix": m.group("space") + m.group("unit"),
                    "aliases": [a for a in group.get("aliases", []) if a]})
    return templates


def _decimals_vary_by_class(groups: list[dict]) -> set[str]:
    """Группы, где число знаков задано по классам оборудования и различается: один шаблон им не годится."""
    varying: set[str] = set()
    for group in groups:
        values = {m.group(1) for rule in group.get("rules", []) for m in _DECIMALS_VALUE_RE.finditer(str(rule))}
        if len(values) > 1:
            varying.add((group.get("attribute") or "").strip().lower())
    return varying


def render_templates(doc_dir: Path, groups: list[dict], name: str) -> dict[str, dict]:
    """Шаблоны вывода чисел (знаки после запятой и суффикс единицы) по атрибутам документа:
    из render_templates.json, таблицы единиц документа и словаря форм значений."""
    templates = read_json(doc_dir, "render_templates.json", {})
    for extra in (_templates_from_units_table(groups), _templates_from_value_forms(groups)):
        for attribute, template in extra.items():
            templates.setdefault(attribute, template)
    varying = _decimals_vary_by_class(groups)
    for attribute in [a for a in templates if a.strip().lower() in varying]:
        templates.pop(attribute, None)
        logger.info("Документ %r: шаблон %r снят — знаки зависят от класса оборудования", name, attribute)
    return templates


# ---------------------------------------------------------------- всё вместе
def legacy_fields(doc_dir: Path, groups: list[dict], name: str, tables: dict[str, DocTable]) -> dict:
    """Поля RuleSet для запасного пути, собранные из файлов извлечения документа."""
    deterministic = read_json(doc_dir, "deterministic_rules.json", {})
    phrase_map = read_json(doc_dir, "phrase_rules.json", {})
    deterministic_order = read_json(doc_dir, "deterministic_order.json", {})
    deterministic_attributes = read_json(doc_dir, "deterministic_attributes.json", {})
    roles = load_table_roles(doc_dir)
    link_cited_lists(roles, groups, tables)

    closed_registers = build_closed_registers(phrase_map, deterministic, set(deterministic_order),
                                              deterministic_attributes)
    attribute_to_register = (attribute_registers_from_origins(
        closed_registers, read_json(doc_dir, "phrase_attributes.json", {}), deterministic_attributes)
        or attribute_registers_from_text(groups, closed_registers))
    closed_registers.update(roles.registers)
    for attribute, register_ids in roles.register_owners.items():
        for register_id in register_ids:
            if register_id not in attribute_to_register.setdefault(attribute, []):
                attribute_to_register[attribute].append(register_id)
    if phrase_map and not closed_registers:
        logger.warning("Документ %r: есть фразовые пары (phrase_rules.json), но ни одна не отнесена "
                       "к закрытому реестру — pending registration не сработает.", name)
    if roles.synonyms:
        logger.info("Документ %r: перечней синонимов %d, атрибутов с ними %d.",
                    name, len(roles.synonyms), len(roles.synonym_owners))

    global_card, cards, card_forms = load_rule_cards(doc_dir)
    examples = [{**e, "group": group} for group, card in cards.items() for e in card.get("examples") or []]
    examples += [{**e, "group": ""} for e in global_card.get("examples") or []]
    return dict(
        deterministic=deterministic, phrase_map=phrase_map, closed_registers=closed_registers,
        attribute_to_register=attribute_to_register, number_first=number_first_default(deterministic_order),
        deterministic_order=deterministic_order, deterministic_attributes=deterministic_attributes,
        excluded_attributes=read_json(doc_dir, "excluded_attributes.json", []),
        render_templates=render_templates(doc_dir, groups, name),
        synonyms=roles.synonyms, synonym_owners=roles.synonym_owners, universal_synonyms=roles.universal,
        global_card=global_card, cards=cards, card_forms=card_forms, examples=examples)
