"""Какое изделие описывает строка: предмет групп, класс изделия, область действия документа."""
import re
from concurrent.futures import ThreadPoolExecutor

from rapidfuzz import fuzz

from ..modules.system_prompts import SUBJECT_CHOICE_PROMPT, SCOPE_PROMPT
from .common import logger


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


def _row_class(rule_set, context: str, model=None) -> str:
    """Класс изделия строки среди классов, которые документ описывает своими разделами.

    Сначала по словам: «Item: Gate valve GV-1101» называет класс «Gate valve» (из нескольких
    подошедших берётся самый длинный — «Pressure relief valve», а не «Valve»). Если класс
    назван кодом или иначе, чем в документе, — один вопрос модели на контекст (кэшируется).
    """
    classes = getattr(rule_set, "class_blocks", None) or {}
    if not classes or not context:
        return ""
    named = [c for c in classes if _subject_matches(context, c)]
    if named:
        return max(named, key=lambda c: len(re.findall(r"\w+", c)))
    if model is None:
        return ""
    return _subject_by_model(model, context, tuple(sorted(classes)))


def _item_named_in_groups(rule_set, context: str) -> list[str]:
    """Группы правил, в имени или алиасах которых названо изделие строки.

    Слова изделия берутся из значений контекста («Item: Flange FLG-5104» → «flange»), без
    подписей колонок и без кодов с цифрами. Смотрятся только имена и алиасы групп, а не весь
    текст правил: слово «flanged» в стандарте на клапаны не делает его стандартом на фланцы.
    """
    values = " ".join(part.split(":", 1)[-1] for part in str(context).split(";"))
    words = [w.lower() for w in re.findall(r"(?<![\w-])[A-Za-z]{4,}(?![\w-])", values)]
    if not words:
        return []
    found = []
    for group in rule_set.groups:
        names = " ".join([group.attribute] + list(group.aliases)).lower()
        name_words = re.findall(r"[a-z]{4,}", names)
        if any(_words_match(w, n) for w in words for n in name_words):
            found.append(group.attribute)
    return found


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
    # у изделия есть свой раздел в документе — документ его описывает, спрашивать не о чем;
    # по одному началу документа модель «громкоговоритель» из области действия исключала
    if _row_class(rule_set, context, model):
        return True
    # изделие названо в именах групп правил («Flange type» для «Flange FLG-5104») — документ его
    # описывает, даже если разметка предметов посчитала эти группы общими. Без этого проверка
    # спрашивала модель по началу документа, где фланцев нет, и строки фланца оставались пустыми
    named = _item_named_in_groups(rule_set, context)
    if named:
        logger.info("  [SCOPE] %r: изделие названо в группах правил %s", context, named[:3])
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
