"""Детерминированный путь строки: исключения, таблицы фраз и синонимов, корни, закрытые реестры."""
import os
import re

from rapidfuzz import fuzz

from ..modules.system_prompts import REGISTER_PROMPT
from ..modules.tables import form_key
from .common import (_attribute_is_excluded, _attribute_matches, _norm_attr, _normalize_phrase,
                     logger)
from .answers import AMBIGUOUS_WORDING, PENDING_WORDING, _is_pending_result


FUZZY_THRESHOLD = 90


# Порог «похоже хоть на что-то»: ниже него реестр даже не обсуждается с моделью.
REGISTER_FLOOR = 40


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


def _has_own_group(rule_set, attribute: str) -> bool:
    """Есть ли в документе группа правил, названная (именем или алиасом) ровно этим атрибутом.

    Тогда реестр или перечень синонимов другого владельца, чьё имя лишь пересекается с подписью
    по словам, не про эту строку: «Insulation» (изоляция кабеля) — не «Insulation class»
    (класс нагревостойкости), «Type» (тип фланца) — не «TEMA type». Без своей группы атрибут,
    наоборот, ищет владельца по словам: имя в документе и подпись в таблице — разные тексты.
    """
    key = _norm_attr(attribute)
    return any(key == _norm_attr(name) for group in getattr(rule_set, "groups", [])
               for name in [group.attribute] + list(group.aliases or []) if name)


def _registers_for_attribute(rule_set, attribute: str) -> list[str]:
    """Префиксы реестров, относящихся к атрибуту строки.

    Имя группы правил в документе и подпись атрибута в таблице заказчика — разные тексты
    («nominal diameter (metric)» против «Nominal size, DN»), поэтому точного совпадения ключа
    мало: доискиваем по вхождению, как и для владельцев keyword-правил.
    """
    exact = rule_set.attribute_to_register.get(attribute.strip().lower())
    if exact:
        return exact
    if _has_own_group(rule_set, attribute):
        return []  # у атрибута свои правила, а реестр «Insulation» — про другой атрибут
    for owner, prefixes in rule_set.attribute_to_register.items():
        if _attribute_matches(attribute, owner):
            return prefixes
    return []


def _synonym_tables_for_attribute(rule_set, attribute: str) -> list[str]:
    """Таблицы синонимов атрибута строки: по точному имени, иначе — всех совпавших владельцев.

    Подпись в таблице заказчика («Roof type») и владелец в документе («tank roof type») — разные
    тексты, поэтому, как и у реестров, сравниваем по целым словам. Совпавших владельцев может
    быть несколько; решает тогда только форма, которая во всех их таблицах ведёт к одному значению.
    """
    owners = getattr(rule_set, "synonym_owners", None) or {}
    exact = owners.get(attribute.strip().lower())
    if exact:
        return list(exact)
    if _has_own_group(rule_set, attribute):
        return []
    found: list[str] = []
    for owner, table_ids in owners.items():
        if _attribute_matches(attribute, owner):
            found.extend(t for t in table_ids if t not in found)
    return found


def _canonical_form(value: str, rule_set, table_ids: list[str]) -> tuple[str, str] | None:
    """(каноническое значение, номер таблицы), если запись — принятая форма из таблиц; иначе None.

    Только точное совпадение формы (без учёта регистра и лишних пробелов): таблица перечисляет
    написания, и «похоже на форму» — уже не форма. Если одна и та же форма в разных таблицах
    ведёт к разным значениям, детерминированно решить нельзя — решает модель по правилам.
    """
    key = form_key(value)
    hits: dict[str, str] = {}
    for table_id in table_ids:
        canonical = rule_set.synonyms.get(table_id, {}).get(key)
        if canonical is not None:
            hits.setdefault(canonical, table_id)
    if len(hits) != 1:
        return None
    return next(iter(hits.items()))


def _apply_synonyms(value: str, rule_set, attribute: str) -> tuple[str, bool, str] | None:
    """Запись — принятая форма из таблицы синонимов документа → её каноническое значение.

    Сначала таблицы, действующие для всех атрибутов (значения «не применимо»/«уточняется»),
    потом таблицы самого атрибута. Значение через «;» — перечисление: каждая позиция отдельно,
    и заменяется, только если заменяются все. Промах — не «pending registration»: таблица
    синонимов не обязана перечислять все значения (номерные обозначения описаны пунктом),
    поэтому такая строка идёт к модели, а таблица — в её промпт.
    """
    if not getattr(rule_set, "synonyms", None):
        return None
    parts = [p.strip() for p in value.split(";") if p.strip()]
    for table_ids in (rule_set.universal_synonyms, _synonym_tables_for_attribute(rule_set, attribute)):
        if not table_ids or not parts:
            continue
        whole = _canonical_form(value, rule_set, table_ids)
        mapped = [whole] if whole else [_canonical_form(p, rule_set, table_ids) for p in parts]
        if all(mapped):
            result = "; ".join(canonical for canonical, _ in mapped)
            tables = ", ".join(dict.fromkeys(t for _, t in mapped))
            return result, False, f"таблица {tables}: «{value.strip()}» → {result}"
    return None


# Карточки правил (rule_cards.json): forms — только принятые формы из них, детерминированно
# (по умолчанию); on — карточки ещё и в промпт к тексту правил; compact — карточки вместо
# сырого текста (кроме группы самого атрибута); off — не используются. В промпте карточки
# вредили: пересказ терял нюансы («2 decimal places» вместо «не больше двух») — MNT 94 → 91,
# EMD 95 → 89; замена форм без модели — 61 строка, ни одной ошибки, MNT 94 → 95.
RULE_CARDS = os.getenv("RULE_CARDS", "forms").strip().lower()


def _apply_card_forms(value: str, rule_set, attribute: str) -> tuple[str, bool, str] | None:
    """Запись — принятая форма из карточки группы атрибута → основная запись этой группы.

    То же, что таблица синонимов, но для документа, где принятые формы стоят прозой («pipe is
    Seamless (‘SMLS’ and ‘S/L’ accepted)»): модель читала «accepted» то как «заменить», то как
    «можно оставить». Карточка решила это один раз при загрузке документа. Владелец — группа по
    имени атрибута (не по алиасам: они слишком широки); несколько групп дают разные записи — не
    решаем, строка идёт к модели.
    """
    forms = getattr(rule_set, "card_forms", None) or {}
    if RULE_CARDS == "off" or not forms:
        return None
    key = attribute.strip().lower()
    owners = [key] if key in forms else [o for o in forms if _attribute_matches(attribute, o)]
    hits = {forms[o][form_key(value)] for o in owners if form_key(value) in forms[o]}
    if len(hits) != 1:
        return None
    written = hits.pop()
    return written, False, f"карточка правил: «{value.strip()}» → {written}"


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

    # 0.5) запись — принятая форма из таблицы синонимов документа
    synonym = _apply_synonyms(result, rule_set, attribute)
    if synonym:
        return synonym

    # 0.6) запись — принятая форма из карточки правил атрибута
    carded = _apply_card_forms(result, rule_set, attribute)
    if carded:
        return carded

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
