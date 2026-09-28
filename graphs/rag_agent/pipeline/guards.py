"""Проверки ответа модели: чего правила не оправдывают, то отклоняется или снимается."""
import re

from ..modules.tools import unit_mismatch, is_unit
from .common import (_NUM_UNIT_RE, _WORD_RE, _alnum, _attribute_matches, _char_bag, _digit_groups,
                     _names_clash, _norm_attr, _unit_key, logger)
from .deterministic import _synonym_tables_for_attribute


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
    # канонические значения таблиц синонимов атрибута: «Bellows» → «Balanced bellows» по таблице
    for table_id in _synonym_tables_for_attribute(rule_set, attribute) + rule_set.universal_synonyms:
        for canonical in rule_set.synonyms.get(table_id, {}).values():
            tokens.update(w.lower() for w in _WORD_RE.findall(str(canonical)))
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
