"""Проверки ответа модели: чего правила не оправдывают, то отклоняется или снимается."""
import re

from ..modules.system_prompts import QUANTITY_KIND_PROMPT
from ..modules.tools import convert_value, is_unit, same_dimension, unit_mismatch
from .common import Memo
from .common import (_WORD_RE, _alnum, _attribute_matches, _char_bag, _context_line,
                     _names_clash, _norm_attr, _unit_key, logger)
from .deterministic import _synonym_tables_for_attribute
from .parsing import Parsed, is_measure, parse, recorded_quantity, recorded_unit, units_differ


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
    # выбираем шаблон, чья единица совпадает с единицей самого значения (разбор модели)
    parsed = parse(value, attribute) if value else None
    quantity = parsed.single if parsed else None
    if quantity:
        for tmpl in candidates:
            if _suffix_is_unit(tmpl.get("suffix", ""), quantity):
                return tmpl
    return candidates[0]


def _suffix_is_unit(suffix: str, quantity) -> bool:
    """Суффикс шаблона — это единица величины: то же написание либо та же единица у pint."""
    suffix = str(suffix).strip()
    if not suffix or not quantity.unit_text:
        return False
    return (_unit_key(suffix) == _unit_key(quantity.unit_text)
            or (is_unit(suffix) and quantity.unit and not unit_mismatch(suffix, quantity.unit)))


def _apply_render_template(answer: str, template: dict) -> str:
    """Детерминированно переформатирует ответ под шаблон (decimals + suffix).

    Шаблон только ПЕРЕрисовывает уже полученное значение и никогда не навешивает единицу:
    применяется, лишь если ответ — одно число со «своим» суффиксом (разбор модели). Иначе
    (диапазон, код, другая единица) ответ не трогаем.
    """
    parsed = parse(answer)
    q = parsed.single if parsed else None
    if q is None or q.value is None or not _suffix_is_unit(template["suffix"], q):
        return answer
    if f"{q.number}{' ' if q.space else ''}{q.unit_text}" != str(answer).strip():
        return answer  # в ответе есть что-то кроме числа с единицей
    return f"{q.value:.{template['decimals']}f}" + template["suffix"]


def _example_keeps_space(rule_set, rules_text: str, root: str) -> bool:
    """Показывает ли пример документа этот корень с пробелом перед числом.

    «Class 150» в примере значит, что склеивать нельзя, как бы ни выглядели корни в других
    документах. Если примера с этим корнем нет, решает общая конвенция keyword→root. Примеры —
    из карточек документа, корень и пробел в них — по разбору модели.
    """
    for _, target in _examples_in(rule_set, rules_text):
        d = (parse(target) or Parsed("text")).designation
        if d and d.space and d.root.upper() == str(root).upper():
            return True
    return False


def _align_root_spacing(answer: str, rule_set, rules_text: str = "") -> str:
    """Корень и число склеиваются так же, как их собирает детерминированный путь: без пробела.

    Один и тот же атрибут давал «TRIM8» в одной строке и «TRIM 12» в другой, «CL600» и
    «CL 600»: конвенцию документа знает keyword→root, а LLM-путь о ней не знает. Правится
    только пробел между корнем и числом, содержимое не трогается. Если же документ показал
    в примере пробел («150 Class» → «Class 150»), его пример и решает. Где в ответе корень и
    номер, говорит разбор модели (parsing.py), а не шаблон.
    """
    roots = {root.upper() for root in rule_set.deterministic.values() if root and root.isalpha()}
    if not roots:
        return answer
    parsed = parse(answer)
    d = parsed.designation if parsed else None
    if d is None or not d.space or d.root.upper() not in roots:
        return answer
    if _example_keeps_space(rule_set, rules_text, d.root):
        return answer
    first, second = (d.number, d.root) if d.number_first else (d.root, d.number)
    aligned = re.sub(rf"({re.escape(first)})\s+({re.escape(second)})", r"\1\2", str(answer), count=1)
    if aligned != answer:
        logger.info("  [ROOT] %r -> %r (конвенция документа: корень и число слитно)",
                    answer, aligned)
    return aligned


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
    слово его ключа. Корень ответа берётся из разбора модели.
    """
    parsed = parse(answer)
    if parsed is None or parsed.designation is None:
        return False
    root = parsed.designation.root.upper()
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
    Номер обозначения целый: «NPS6.000» — это пересчитанная длина. Что корень, а что номер,
    говорит разбор модели.
    """
    roots = {r.upper() for r in rule_set.deterministic.values() if r and r.isalpha() and len(r) > 1}
    if not roots:
        return False
    parsed = parse(answer)
    if parsed is None:
        return False
    d = parsed.designation
    if d is not None and d.root.upper() in roots:
        number = d.number.strip()
        return not number[:1].isdigit() or bool(re.match(r"^\d+[.,]\d", number))
    # корень стоит в начале, а ответ модель прочла как величину: корень навешен на измерение
    head = str(answer).strip().upper()
    if parsed.quantities and any(head.startswith(r) for r in roots):
        return bool(recorded_unit(parse(value)))
    return False


def _unit_invented(attribute: str, value: str, answer: str) -> bool:
    """Единица появилась из ниоткуда: её нет ни в записи, ни в подписи атрибута.

    «150» в графе «Nominal size» превращалось в «150.000 in.» — это уже утверждение о величине,
    а не о её написании. Документ вправе требовать единицу, но взять её неоткуда: решает человек.
    Что в записи голое число и что подпись единицы не называет, говорит разбор модели.
    """
    answer_parsed = parse(answer)
    if answer_parsed is None or not answer_parsed.single or not is_measure(answer_parsed.single.unit):
        return False  # ответ без единицы — судить не о чем
    recorded = parse(value, attribute)
    return recorded is not None and recorded.kind == "number" and not recorded.label_unit


def _unit_relabelled(attribute: str, value: str, answer: str) -> bool:
    """Единица подменена без пересчёта: число то же, а единица уже другая.

    «Wall thickness, mm» со значением «9.5» превращалось в «9.50 in.» — запись просто
    переобозначили дюймами. Если пересчёт был, меняется число; если не был, единица обязана
    остаться той, что названа в записи или в подписи атрибута. Единицы сравнивает pint по
    стандартным именам из разбора, отметку давления — разбор модели: «in.» и «inch» — одно,
    «barg» и «bara» — нет, «barg» и «bar(g)» — одно.
    """
    answer_parsed = parse(answer)
    now = answer_parsed.single if answer_parsed else None
    was = recorded_quantity(parse(value, attribute))
    if now is None or was is None or not now.unit or not units_differ(was, now):
        return False
    return was.value is not None and now.value is not None and abs(was.value - now.value) < 1e-9


def _document_mandates_spacing(rule_set, group_names: list[str] | None = None) -> bool:
    """Решает ли документ сам, ставить ли пробел между числом и единицей.

    Ответ берётся из карточек (поле unit_spacing): общих положений и групп этой строки. Раньше
    его искали в тексте правил по оборотам вроде «one space» — и пропускали любую другую
    формулировку, в том числе требование, показанное только примером.
    """
    cards = getattr(rule_set, "cards", None) or {}
    if (getattr(rule_set, "global_card", None) or {}).get("unit_spacing"):
        return True
    return any((cards.get(name) or {}).get("unit_spacing") for name in group_names or [])


def _restore_unit_spacing(value: str, answer: str, attribute: str = "") -> str:
    """Пробел между числом и единицей берётся из записи, если единица та же.

    Документ задаёт единицу и знаки после запятой, но молчит про пробел; модель пишет
    «16.00mm2» там, где записано «16 mm2». Где правила молчат, формат записи и сохраняем.
    Что число, что единица и был ли между ними пробел, говорит разбор модели.
    """
    recorded, parsed = parse(value, attribute), parse(answer)
    if recorded is None or parsed is None or not recorded.single or not parsed.single:
        return answer
    was, now = recorded.single, parsed.single
    if not was.unit_text or not now.unit_text or not is_measure(now.unit):
        return answer  # «- GP - 2» из номера модели — не единица, пробелы там не наши
    # пробел переносим только при том же написании единицы: у «0.5-inch» пробела нет, но
    # «0.500 in.» — уже другое написание, и склеивать «0.500in.» значит испортить ответ
    if (unit_mismatch(was.unit, now.unit) or _unit_key(was.unit_text) != _unit_key(now.unit_text)
            or was.space == now.space):
        return answer
    if f"{now.number}{' ' if now.space else ''}{now.unit_text}" != str(answer).strip():
        return answer  # ответ не состоит из одного числа с единицей — не перекраиваем
    return f"{now.number}{' ' if was.space else ''}{now.unit_text}"


def _document_word_tokens(rule_set, attribute: str, group_names: list[str] | None = None) -> set:
    """Слова, которые документ вправе внести в ответ: корни и коды его таблиц."""
    tokens = set()
    for root in rule_set.deterministic.values():
        tokens.update(w.lower() for w in _WORD_RE.findall(str(root)))
    for table in rule_set.closed_registers.values():
        for code in table.values():
            tokens.update(w.lower() for w in _WORD_RE.findall(str(code)))
    # канонические значения таблиц синонимов атрибута: «Bellows» → «Balanced bellows» по таблице
    for table_id in _synonym_tables_for_attribute(rule_set, attribute) + rule_set.universal_synonyms:
        for canonical in rule_set.synonyms.get(table_id, {}).values():
            tokens.update(w.lower() for w in _WORD_RE.findall(str(canonical)))
    template = _match_render_template(attribute, rule_set.render_templates, group_names)
    if template:
        tokens.update(w.lower() for w in _WORD_RE.findall(str(template.get("suffix", ""))))
    return tokens


def _relabel_sanctioned(value: str, answer: str, rule_set, rules_text: str,
                        attribute: str = "") -> bool:
    """Показывает ли документ ту же смену единицы при том же числе своим примером.

    Разность температур пишется в K тем же числом, что и в °C («5 °C» → «5 K»): это не подмена,
    а правило, и пример документа говорит об этом прямо. Примеры — из карточек, числа и единицы
    в них — по разбору модели.
    """
    was = recorded_quantity(parse(value, attribute))
    answer_parsed = parse(answer)
    now = answer_parsed.single if answer_parsed else None
    if was is None or now is None:
        return False
    for source, target in _examples_in(rule_set, rules_text):
        s_parsed, t_parsed = parse(source), parse(target)
        s_q = s_parsed.single if s_parsed else None
        t_q = t_parsed.single if t_parsed else None
        if (s_q and t_q and s_q.value is not None and s_q.value == t_q.value
                and not units_differ(s_q, was) and not units_differ(t_q, now)):
            return True
    return False


def _norm_text(text: str) -> str:
    """Для поиска примера в тексте правил: регистр, кавычки и пробелы не важны."""
    return re.sub(r"\s+", " ", re.sub(r"[‘’“”'\"]", "", str(text).lower())).strip()


def _examples_in(rule_set, rules_text: str) -> list[tuple[str, str]]:
    """Примеры «было → стало» из карточек документа, которые стоят в этом тексте правил.

    Раньше примеры выискивались в тексте регулярками (EXAMPLE, стрелки, raw/normalized record,
    ячейки образца датащита) — и пропускали любую другую подачу. Теперь их один раз при
    загрузке выписывает модель в карточки, а код берёт те, что попали в промпт строки.
    """
    examples = getattr(rule_set, "examples", None) or []
    if not examples or not rules_text:
        return []
    haystack = _norm_text(rules_text)
    return [(e["from"], e["to"]) for e in examples
            if _norm_text(e["from"]) in haystack and _norm_text(e["to"]) in haystack]


def _example_leak(value: str, answer: str, rules_text: str, allowed: set = frozenset(),
                  examples: list[tuple[str, str]] | None = None) -> list[str]:
    """Слова ответа, взятые из записи-примера документа, а не из этой записи.

    «Dependent, pressure» превращалось в «Pressure independent»: пример документа —
    «Independent, pressure» → «Pressure independent», и модель перенесла из него слово. Слово
    из исходной стороны примера — это содержимое чужой записи; в ответ оно может попасть, только
    если есть и в этой записи. Исключение — пример, чья исходная запись целиком входит в эту
    («Al» → «Aluminium» для записи «Al»): тогда его результат и есть предписанный ответ.
    Примеры (обе стороны) — из карточек документа (_examples_in).
    """
    # слова — буквенные цепочки: в «DN32» слово «DN» есть, и «DN 32» из записи не заимствует
    words = lambda text: {w.lower() for w in re.findall(r"[^\W\d_]+", str(text)) if len(w) > 1}
    value_words = words(value)
    pairs = examples or []
    licensed: set = set()
    for left, right in pairs:
        if words(left) and words(left) <= value_words:
            licensed |= words(right)
    foreign = set().union(*[words(side) for pair in pairs for side in pair]) if pairs else set()
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
            and not _relabel_sanctioned(value, answer, rule_set, rules_text, attribute)):
        return "единица подменена без пересчёта — оставлено как записано"
    if _root_invented(value, answer, rule_set, attribute):
        return "запись не называет это обозначение — оставлено как записано"
    leak = _example_leak(value, answer, rules_text, _document_word_tokens(rule_set, attribute),
                         _examples_in(rule_set, rules_text))
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


_QUANTITY_KIND = Memo("род величины атрибута")


def _records_quantity(model, attribute: str, value: str, unit: str, context: str = "") -> bool:
    """Записывает ли атрибут величину того же рода, что `unit`, а не класс/обозначение.

    По единице значения этого не понять: «Inlet rating: 300 lbs» — класс давления, а читается
    как масса. Внутри нормализации модель на этот вопрос отвечала плохо (переводила класс в
    тонны), отдельным вопросом — уверенно. Один вопрос на пару (атрибут, единица).
    """
    key = (str(attribute).strip().lower(), str(unit).strip().lower(), str(context).strip().lower())
    return _QUANTITY_KIND.remember(key, lambda: _ask_quantity_kind(model, attribute, value, unit, context))


def _ask_quantity_kind(model, attribute: str, value: str, unit: str, context: str) -> bool:
    """Один вопрос модели (QUANTITY_KIND_PROMPT): величина ли это рода `unit`; без ответа — да."""
    messages = [
        {"role": "system", "content": QUANTITY_KIND_PROMPT},
        {"role": "user", "content": (f"{_context_line(context)}Attribute: {attribute}\n"
                                     f"Recorded value: {value}\nUnit: {unit}")},
    ]
    try:
        answer = str(model.invoke(messages).content).strip().upper()
    except Exception as e:
        logger.warning("  [QUANTITY] вопрос о роде величины не удался: %s", e)
        answer = "YES"  # по единице величина та же — без ответа модели этого и держимся
    verdict = not answer.startswith("NO")
    logger.info("  [QUANTITY] %r (%r) — величина рода %r: %s", attribute, value, unit, verdict)
    return verdict


def _conversion_of_designation(attribute: str, value: str, answer: str, model,
                               context: str = "") -> str:
    """Пересчитала ли модель в другую единицу то, что величиной не является.

    Класс давления пишут «в фунтах» («Rating: 800 lbs»), и модель честно переводила его в
    килограммы — 362.87 kg. По одной записи класс от массы не отличить, это видно по атрибуту,
    поэтому, только когда ответ — пересчёт, модель один раз на атрибут отвечает, величина ли он.
    Пересчёт ли это (та же размерность, другая единица), решает pint по разбору модели.
    Пустая строка — пересчёт законный или его не было.
    """
    if model is None:
        return ""
    recorded, parsed = parse(value, attribute), parse(answer)
    if recorded is None or parsed is None or not recorded.single or not parsed.single:
        return ""
    was, now = recorded.single, parsed.single
    if not (was.unit and now.unit and same_dimension(was.unit, now.unit)
            and unit_mismatch(was.unit, now.unit)):  # отметка давления — не пересчёт
        return ""
    if _records_quantity(model, attribute, value, was.unit_text or was.unit, context):
        return ""
    return ("атрибут — класс или обозначение, а не величина: пересчёт единиц отменён — "
            "оставлено как записано")


CONVERSION_TOLERANCE = 0.01  # относительная; номинальные размеры («4 in» → «100 mm») на 1.6% — не ошибка счёта


def _decimals_of(number: str) -> int:
    """Знаков после запятой в числе как записано: «3.50» → 2, «1 000,5» → 1."""
    text = str(number).strip().replace(",", ".")
    return len(text.rsplit(".", 1)[1]) if "." in text else 0


def _conversion_arithmetic(attribute: str, value: str, answer: str) -> str:
    """Сходится ли число в ответе с пересчётом записи: «350 kPag» → «35 barg» не сходится.

    Модель то вызывает инструмент пересчёта, то считает в уме и ошибается на порядок, а ни одна
    другая проверка этого не видит: единицы в ответе законные. Считает pint по разбору записи
    и ответа; расхождение больше округления ответа и CONVERSION_TOLERANCE — повод показать
    строку человеку (ответ не откатывается: правило документа могло велеть округлить иначе).
    Пустая строка — пересчёта нет или он верный. Отметка давления у сторон разная — не судим.
    """
    recorded, parsed = recorded_quantity(parse(value, attribute)), parse(answer)
    now = parsed.single if parsed else None
    if recorded is None or now is None or recorded.value is None or now.value is None:
        return ""
    if not (recorded.unit and now.unit and same_dimension(recorded.unit, now.unit)
            and unit_mismatch(recorded.unit, now.unit)):
        return ""  # единица та же или это не пересчёт
    if recorded.reference != now.reference:
        return ""  # манометрическое ↔ абсолютное: к пересчёту добавляется атмосфера
    expected = convert_value(recorded.value, recorded.unit, now.unit)
    if expected is None:
        return ""
    allowed = max(abs(expected) * CONVERSION_TOLERANCE, 0.5 * 10 ** -_decimals_of(now.number))
    if abs(expected - now.value) <= allowed:
        return ""
    return (f"число не сходится с пересчётом: {recorded.number} {recorded.unit} = "
            f"{expected:.6g} {now.unit}, в ответе {now.number} — проверьте вручную")


# Число с нулями в конце дробной части: «20.0», «5.50», «0.00». Не трогаем число внутри
# обозначения («B16.10», «2.50X»): перед ним буква или цифра, после — буква.
_TRAILING_ZERO_RE = re.compile(r"(?<![A-Za-z\d.])(\d+)\.(\d*?)0+(?=\s|$|[^\w.])")


def _drop_trailing_zeros(answer: str) -> str:
    """«20.0 mbarg» → «20 mbarg», «5.50 m» → «5.5 m», «0.00 to 0.25 bar» → «0 to 0.25 bar».

    Только для документа, чьи общие положения прямо велят убирать такие нули (поле
    trailing_zeros карточки). Модель читала «округлить до одного знака» как «показать один знак»
    и писала «50.0 kg»; документ с фиксированным числом знаков («0.630 in.») сюда не попадает.
    """
    return _TRAILING_ZERO_RE.sub(lambda m: m.group(1) + (f".{m.group(2)}" if m.group(2) else ""),
                                 str(answer))
