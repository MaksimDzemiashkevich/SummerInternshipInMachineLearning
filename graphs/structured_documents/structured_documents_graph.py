import json
import os
import re

from graphs.structured_documents.block_grouping import build_units, group_units, parse_page_content
from graphs.structured_documents.modules.config_structured_documents_graph import Settings
from dotenv import load_dotenv

load_dotenv()
settings = Settings()

# Кэши привязаны к конкретному документу, чтобы несколько PDF можно было
# экстрактить последовательно и накапливать правила в rule_groups.json.
_doc_stem = settings.input_path_document.stem
_out_dir = settings.RULES_STORAGE_PATH.parent
_doc_dir = _out_dir / _doc_stem
_pages_cache = _doc_dir / "pages_data.json"
_pages_partial = _doc_dir / "pages_data.partial.json"
_check_cache = _doc_dir / "data_after_check.json"
_rules_per_doc_path = _doc_dir / "rule_groups.json"
_deterministic_path = _doc_dir / "deterministic_rules.json"
_deterministic_order_path = _doc_dir / "deterministic_order.json"
_deterministic_attributes_path = _doc_dir / "deterministic_attributes.json"
_phrase_path = _doc_dir / "phrase_rules.json"
_phrase_attributes_path = _doc_dir / "phrase_attributes.json"
_excluded_path = _doc_dir / "excluded_attributes.json"
_render_path = _doc_dir / "render_templates.json"

import base64
import fitz
from langchain_openai import ChatOpenAI

from graphs.structured_documents.subject_tagging import tag_subjects
from graphs.structured_documents.rule_cards import compile_cards
from graphs.structured_documents.table_roles import classify_lists_voted
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langgraph.graph import END, START, StateGraph, MessagesState
from PIL import Image as PILImage
import io
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import pytesseract
from pydantic import BaseModel, Field
from pathlib import Path


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


vlm = ChatOpenAI(
    model=settings.VL_model,
    base_url=settings.VL_url,
    api_key=settings.VL_api_key,
    max_tokens=settings.VL_max_tokens,
)

class MessagesState(MessagesState):
    pages_data : list[dict]
    data_after_check: list[dict]
    rule_groups: list[dict]

class RuleGroup(BaseModel):
    attribute: str = Field(description="Concrete attribute the rules apply to (e.g. 'trim', 'gate', 'end connection'); use 'Global' for rules that apply to all attributes")
    type: str = Field(default="local", description="'global' if the rule applies to all attributes, 'local' if it applies to one specific attribute")
    aliases: list[str] = Field(default_factory=list, description="Aliases/synonyms/terms this group typically covers (oil&gas terminology)")
    rules: list[str] = Field(description="List of rules/requirements in this group")

class GroupingResult(BaseModel):
    groups: list[RuleGroup] = Field(description="List of all found rule groups by attribute")

class ParentGroup(BaseModel):
    name: str = Field(description="Short parent theme name (main noun)")
    attributes: list[str] = Field(description="Attribute names that belong to this parent")

class ClusteringResult(BaseModel):
    parents: list[ParentGroup] = Field(description="List of parent groups, each with its attributes")

def load_json(path: Path):
    """Загружает JSON с диска. Возвращает None, если файла нет или он битый."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"WARNING: не удалось загрузить {path}: {e}")
        return None


def save_json(obj, path: Path) -> None:
    """Сохраняет объект в JSON на диск (создавая родительские каталоги при необходимости)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    print(f"Сохранено в {path}")


VLM_ATTEMPTS = 8
VLM_WAIT_SECONDS = 60


def _invoke_vlm(messages, page_number: int):
    """Вызов VLM с повторами: короткий обрыв туннеля не должен останавливать документ."""
    for attempt in range(1, VLM_ATTEMPTS + 1):
        try:
            return vlm.invoke(messages)
        except Exception as e:
            if attempt == VLM_ATTEMPTS:
                raise
            print(f"VLM недоступна (стр. {page_number + 1}, попытка {attempt}/{VLM_ATTEMPTS}): "
                  f"{type(e).__name__}; жду {VLM_WAIT_SECONDS * attempt} с")
            time.sleep(VLM_WAIT_SECONDS * attempt)


# Сколько страниц распознаётся одновременно. Страницы независимы, а vLLM обслуживает несколько
# запросов сразу (непрерывный батчинг): при --max-num-seqs 1 на сервере две карты простаивали
# на декодировании одной страницы. Число не должно быть больше --max-num-seqs сервера.
VLM_WORKERS = int(os.getenv("VLM_WORKERS", "4"))


def _read_page(document, page_number: int) -> dict:
    """Одна страница: картинка, поворот по OSD, ответ VLM — запись для pages_data.json."""
    pix = document[page_number].get_pixmap(dpi=160)
    img = PILImage.open(io.BytesIO(pix.tobytes("png")))

    detected_angle = 0
    confidence = 0.0
    try:
        osd = pytesseract.image_to_osd(img)
        detected_angle = int(
            [line for line in osd.split("\n") if "Rotate" in line][0].split(":")[1].strip()
        )
        confidence = float(
            [line for line in osd.split("\n") if "Orientation confidence" in line][0].split(":")[1].strip()
        )
    except pytesseract.TesseractError as e:
        print(f"OSD не смог определить ориентацию (стр. {page_number + 1}): {e}")

    if detected_angle != 0 and confidence >= 1.0:
        img = img.rotate(-detected_angle, expand=True)
        print(f"Стр. {page_number + 1}: изображение повёрнуто на {-detected_angle}°")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    message = HumanMessage(content=[
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
        {"type": "text", "text": "Extract text from this page."}
    ])
    response = _invoke_vlm([SystemMessage(content=settings.VL_prompt), message], page_number)
    token_usage = response.response_metadata.get("token_usage")
    return {
        "page_number": page_number,
        "source_page_index": page_number,
        "detected_angle": detected_angle,
        "confidence": confidence,
        "content": response.content,
        "token_usage": token_usage,
    }


def first_node(state: MessagesState) -> MessagesState:
    cached = load_json(_pages_cache)
    if cached is not None:
        print(f"Загружено {len(cached)} страниц из кэша {_pages_cache}")
        return {"messages": [], "pages_data": cached}

    _document = fitz.open(settings.input_path_document)
    total = len(_document)

    # Распознанные страницы сохраняются по одной: туннель к VLM рвётся посреди длинного документа,
    # и обрыв на 80-й странице не должен выбрасывать 79 готовых — повторный запуск продолжит с места.
    # Страницы идут параллельно и заканчиваются в любом порядке, поэтому продолжение пропускает
    # готовые номера, а не первые N записей.
    pages_data = load_json(_pages_partial) or []
    done = {page["page_number"] for page in pages_data}
    todo = [n for n in range(total) if n not in done]
    if pages_data:
        print(f"Продолжаю распознавание: готово {len(done)} из {total} страниц")

    lock = threading.Lock()

    def work(page_number: int) -> None:
        # PyMuPDF не потокобезопасен на одном документе: у каждого потока свой
        page = _read_page(fitz.open(settings.input_path_document), page_number)
        with lock:
            pages_data.append(page)
            save_json(pages_data, _pages_partial)
            print(f"Стр. {page_number + 1} готова ({len(pages_data)}/{total}), токены: {page['token_usage']}")

    with ThreadPoolExecutor(max_workers=VLM_WORKERS) as pool:
        for future in [pool.submit(work, n) for n in todo]:
            future.result()  # ошибка страницы (туннель лёг насовсем) останавливает прогон; готовое сохранено

    pages_data.sort(key=lambda page: page["page_number"])
    save_json(pages_data, _pages_cache)
    _pages_partial.unlink(missing_ok=True)

    return {"messages": [], "pages_data": pages_data}

def formatting_text_node(state: MessagesState) -> MessagesState:
    cached = load_json(_check_cache)
    if cached is not None:
        print(f"Загружено {len(cached)} страниц после форматирования из кэша {_check_cache}")
        return {"messages": [], "data_after_check": cached}

    llm = ChatOpenAI(model=settings.llmodel, model_kwargs={"response_format": {"type": "json_object"}})

    pages = state.get("pages_data")

    def prepared(page: dict) -> dict:
        page_copy = dict(page)
        content = parse_page_content(page.get("content"))
        if content is None:
            print(f"WARNING: страница {page.get('page_number')}: ответ VLM не разбирается как JSON, "
                  f"проверка получит сырой текст")
        else:
            page_copy["content"] = content
        return page_copy

    def check(window: list[dict]) -> dict:
        """Проверка первой страницы окна (вторая — только контекст для переноса абзацев)."""
        own = prepared(window[0])
        result = llm.invoke([
            SystemMessage(content=settings.llm_prompt),
            HumanMessage(content=json.dumps([prepared(p) for p in window], ensure_ascii=False)),
        ])
        print(f"page {own.get('page_number')}: {result.response_metadata.get('token_usage')}")
        try:
            result_page = json.loads(result.content)["pages"][0]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as e:
            result_page, content = None, None
            print(f"WARNING: страница {own.get('page_number')}: ответ проверки не разобран ({e})")
        else:
            content = parse_page_content(result_page.get("content"))
        # Проверка иногда обрывает JSON страницы (не закрыт массив блоков) — такую страницу
        # группировка потеряла бы целиком, поэтому остаётся страница VLM, а не ответ проверки.
        if content is None or not content.get("blocks"):
            print(f"WARNING: страница {own.get('page_number')}: проверка вернула негодную страницу, "
                  f"оставлена страница VLM")
            return own
        result_page["content"] = content
        return result_page

    # окно из двух страниц — чтобы склеить абзац, разорванный границей; последняя идёт одна
    pages_after_llm = [check(pages[i:i + 2]) for i in range(len(pages))]

    # print(f"\n\ndata_after_check {pages_after_llm}\n\n")

    save_json(pages_after_llm, _check_cache)

    return {
        "messages": [AIMessage(content=json.dumps(pages_after_llm, ensure_ascii=False))],
        "data_after_check": pages_after_llm,
    }


grouping_llm = ChatOpenAI(
    model=settings.llmodel,
    max_tokens=16000,
    temperature=0,
)

# «blocks» — весь документ разом, модель назначает блоки группам (block_grouping.py);
# «pages» — прежний режим: модель переписывает правила каждой страницы в группы.
GROUPING_MODE = os.getenv("GROUPING_MODE", "blocks")

clustering_llm = ChatOpenAI(
    model=settings.llmodel,
    max_tokens=10000,
    temperature=0,
)

def grouping_rules_node(state: MessagesState) -> MessagesState:
    source_data = state.get("data_after_check") or state.get("pages_data")

    # Манифест: какие документы уже обработаны (каждый хранит свои правила отдельно).
    processed_docs = load_json(settings.PROCESSED_DOCS_PATH) or []
    if _doc_stem in processed_docs:
        existing = load_json(_rules_per_doc_path) or []
        print(f"Документ {_doc_stem!r} уже обработан — пропускаем группировку ({len(existing)} групп)")
        return {"rule_groups": existing}

    # Правила каждого документа хранятся отдельно — начинаем с нуля для этого документа.
    all_groups: list[dict] = []
    known_attributes: list[str] = []

    def prepare(page):
        page_copy = dict(page)
        content = page.get("content")
        if isinstance(content, str):
            try:
                page_copy["content"] = json.loads(content)
            except json.JSONDecodeError:
                pass
        return page_copy

    def call_llm(pages_chunk):
        nonlocal all_groups, known_attributes

        hint = ""
        if known_attributes:
            hint = (
                "\n\n=== УЖЕ СУЩЕСТВУЮЩИЕ ГРУППЫ ===\n"
                "Уже есть такие группы:\n- "
                + "\n- ".join(known_attributes)
                + "\nИспользуй существующее название ТОЛЬКО если тема ЯВНО совпадает. Если сомневаешься — создай новую группу с точным названием атрибута."
            )

        full_prompt = settings.TABLE_EXTRACTION_PROMPT + hint  # + JSON_SCHEMA_INSTRUCTION

        message = HumanMessage(content=[
            {"type": "text", "text": json.dumps(pages_chunk, ensure_ascii=False)}
        ])

        print(f"to the grouping rules model:\n{json.dumps(pages_chunk, ensure_ascii=False)}\n")

        response = grouping_llm.invoke([SystemMessage(content=full_prompt), message])
        print(f"\ngrouping tokens: {response.response_metadata.get('token_usage')}")

        print(f"\nresponse llm:\n{response.content}\n")
        try:
            parsed = json.loads(response.content)
            result = GroupingResult.model_validate(parsed)
        except (json.JSONDecodeError, Exception) as e:
            print("PARSING ERROR:", e)
            print("RAW OUTPUT:", response.content)
            raise

        for g in result.groups:
            g_dict = g.model_dump()
            attr = g_dict["attribute"]
            typ = g_dict.get("type", "local")
            merged = False
            for grp in all_groups:
                if grp.get("attribute") == attr and grp.get("type") == typ:
                    existing_rules = grp.setdefault("rules", [])
                    for r in g_dict.get("rules", []):
                        if r not in existing_rules:
                            existing_rules.append(r)
                    merged = True
                    break
            if not merged:
                all_groups.append(g_dict)
            if attr not in known_attributes:
                known_attributes.append(attr)

    if GROUPING_MODE == "blocks":
        # модель раскладывает номера блоков, текст правил собирается дословно (block_grouping.py)
        units = build_units(source_data)
        all_groups, lost = group_units(units, grouping_llm, settings.BLOCK_GROUPING_PROMPT)
        print(f"Блоков: {len(units)}, групп: {len(all_groups)}")
        for unit in lost:
            print(f"WARNING: блок {unit['id']} не попал ни в одну группу: {unit['text'][:120]!r}")
    elif len(source_data) == 1:
        call_llm([prepare(source_data[0])])
    else:
        for i in range(len(source_data)):
            call_llm([prepare(source_data[i])])

    # --- вычищаем детерминированные/фразовые правила в отдельные файлы и убираем пустые группы ---
    det, det_order, det_attrs = extract_deterministic_rules(all_groups)
    save_json(det, _deterministic_path)
    save_json(det_order, _deterministic_order_path)
    save_json(det_attrs, _deterministic_attributes_path)
    phrase_map, phrase_attributes = extract_phrase_rules(all_groups)
    save_json(phrase_map, _phrase_path)
    save_json(phrase_attributes, _phrase_attributes_path)
    save_json(extract_excluded_attributes(all_groups), _excluded_path)
    save_json(extract_render_templates(all_groups), _render_path)
    all_groups = drop_empty_groups(all_groups)

    # --- сохраняем правила этого документа в отдельный файл и помечаем документ обработанным ---
    save_json(all_groups, _rules_per_doc_path)
    processed_docs.append(_doc_stem)
    save_json(processed_docs, settings.PROCESSED_DOCS_PATH)

    return {"rule_groups": all_groups}


# Захват только содержимого кавычек: root обрывается на закрывающей кавычке, а не на
# пунктуации после неё (иначе worked-example «(DN 80 → DN80)» попадал в корень).
_DET_RE = re.compile(r"Keyword:\s*['\"](.+?)['\"]\s*;\s*Invented Root:\s*['\"](.+?)['\"]")


def extract_deterministic_rules(rule_groups: list[dict]) -> tuple[dict[str, str], dict[str, bool], dict[str, str]]:
    """Извлекает детерминированные «Keyword → Root», порядок конкатенации и атрибут-владельца.

    Порядок вычисляется программно из worked-example («Class 300 → 300CLASSATUM» — число
    первым; «Class 300 → CL300» — корень первым), а не по фиксированной фразе. Атрибут-владелец
    сохраняется для attribute-aware матчинга в apply_deterministic. Вычищает строки из групп.
    """
    deterministic: dict[str, str] = {}
    order: dict[str, bool] = {}
    attributes: dict[str, str] = {}
    for g in rule_groups:
        if g.get("type") == "global":
            continue
        kept_rules = []
        for r in g.get("rules", []):
            m = _DET_RE.search(r)
            if m:
                keyword = m.group(1).strip().strip("'\"")
                root = m.group(2).strip().strip("'\"")
                deterministic[keyword] = root
                attributes[keyword] = g["attribute"]
                # worked-example: «… → 300CLASSATUM» (цифра первым) / «… → CL300» (буква первым)
                am = re.search(r"→\s*[‘’“”'\"]?(\d|[A-Za-z])", r)
                if am:
                    order[keyword] = am.group(1).isdigit()
            else:
                kept_rules.append(r)
        g["rules"] = kept_rules
    return deterministic, order, attributes


_PHRASE_RE = re.compile(r"^\s*(?P<phrase>.+?)\s*→\s*(?P<code>[A-Za-z][A-Za-z0-9\-/]*)\s*$")


# «assigned codes: ECC-301 Threaded end (NPT)» — код + фраза + вариант-в-скобках (Задача F).
_ASSIGNED_CODES_RE = re.compile(r"([A-Za-z][A-Za-z0-9]*-\d+)\s+([^;()]+?)\s*\(([A-Za-z0-9\-/]+)\)")


def extract_phrase_rules(rule_groups: list[dict]) -> tuple[dict[str, str], dict[str, str]]:
    """Извлекает «фраза → код» и происхождение каждой фразы (attribute-владелец).

    Дополнительно (Задача F): если рядом с кодом в таблице указан вариант в скобках
    («assigned codes: ECC-301 Threaded end (NPT)»), скобочная часть становится алиасом:
    phrase_map["Threaded end (NPT)"] = "ECC-301" наравне с phrase_map["Threaded end"].
    """
    phrase_map: dict[str, str] = {}
    phrase_attributes: dict[str, str] = {}
    for g in rule_groups:
        if g.get("type") == "global":
            continue
        for r in g.get("rules", []):
            m = _PHRASE_RE.match(r.strip())
            if (m and " " in m.group("phrase") and ":" not in m.group("phrase")
                    and not re.search(r"\d", m.group("phrase"))):
                phrase = m.group("phrase").strip()
                phrase_map[phrase] = m.group("code")
                phrase_attributes[phrase] = g["attribute"]
            for am in _ASSIGNED_CODES_RE.finditer(r):
                code = am.group(1)
                phrase = am.group(2).strip()
                variant = am.group(3)
                if phrase and " " in phrase:
                    alias = f"{phrase} ({variant})"
                    phrase_map[alias] = code
                    phrase_attributes[alias] = g["attribute"]
    return phrase_map, phrase_attributes


_EXCLUSION_MARKERS = (
    "reproduced exactly",
    "excluded from reformatting",
    "reproduced unchanged",
    "not subject to any presentation rule",
    "not governed by any presentation rule",
)


def extract_excluded_attributes(rule_groups: list[dict]) -> list[str]:
    """Извлекает атрибуты/категории, исключённые из реформатирования, по generic-оборотам.

    В global-группе такой оборот обычно сопровождается поимённым списком (маркеры/переносы
    строк) — каждый пункт списка попадает в исключения. У local-группы флагуется только
    явное «excluded from reformatting» (не «reproduced exactly», которое часто описывает
    лишь подслучай вроде «no value recorded» / «composite trim» / «unmatched gate»), и не
    кросс-ссылки «…under XYZ STD …, shall under this standard be replaced».
    """
    excluded: list[str] = []
    for g in rule_groups:
        for r in g.get("rules", []):
            rl = r.lower()
            if g.get("type") == "global":
                # только списки (с маркерами/переносами), а не одиночные предложения
                if any(m in rl for m in _EXCLUSION_MARKERS) and re.search(r"[•–\n]", r):
                    excluded.extend(_split_exclusion_items(r))
            else:
                if "excluded from reformat" in rl and "shall under this standard" not in rl:
                    excluded.append(g["attribute"])
    return excluded


def _split_exclusion_items(text: str) -> list[str]:
    """Режет global-список исключений на поимённые пункты (маркеры списка/переносы строк)."""
    items: list[str] = []
    for part in re.split(r"[•–\n]", text):
        p = part.strip(" \t\r-–—:;.")
        if not p:
            continue
        low = p.lower()
        if low.startswith(("the following", "since", "without", "these", "no clause", "a national", "free-text")):
            continue
        items.append(p)
    return items


_RENDER_EX_RE = re.compile(r"→\s*[‘’“”'\"](\d+(?:\.\d+)?)([^‘’“”'\"]*)[‘’“”'\"]")


def extract_render_templates(rule_groups: list[dict]) -> dict[str, dict]:
    """Извлекает шаблон рендера (decimals + suffix) из worked-example «X → Y».

    Берётся только пример с единицей измерения (suffix начинается с пробела или содержит °),
    а не keyword→root пример («300CLASSATUM», «CL300» — без пробела/градуса).
    """
    templates: dict[str, dict] = {}
    for g in rule_groups:
        if g.get("type") == "global":
            continue
        for r in g.get("rules", []):
            m = _RENDER_EX_RE.search(r)
            if m:
                num = m.group(1)
                suffix = m.group(2).rstrip()
                if suffix and (suffix[0].isspace() or "°" in suffix):
                    decimals = len(num.split(".")[1]) if "." in num else 0
                    templates[g["attribute"]] = {
                        "decimals": decimals,
                        "suffix": suffix,
                        "aliases": list(g.get("aliases", [])),
                    }
                    break
    return templates


def drop_empty_groups(rule_groups: list[dict]) -> list[dict]:
    """Убирает local-группы, у которых после вычистки не осталось правил."""
    return [
        g for g in rule_groups
        if g.get("type") == "global" or g.get("rules")
    ]


def cluster_groups_node(state: MessagesState) -> MessagesState:
    rule_groups = state.get("rule_groups") or []

    local_groups = [g for g in rule_groups if g.get("type") != "global"]

    if not local_groups:
        return {"rule_groups": rule_groups}

    attr_names = [g["attribute"] for g in local_groups]
    try:
        response = clustering_llm.invoke([
            SystemMessage(content=settings.CLUSTERING_PROMPT),
            HumanMessage(content=json.dumps(attr_names, ensure_ascii=False)),
        ])
        parsed = json.loads(response.content)
        result = ClusteringResult.model_validate(parsed)
    except Exception as e:
        print("CLUSTERING ERROR (falling back, no parent):", type(e).__name__, str(e)[:200])
        return {"rule_groups": rule_groups}

    attr_to_parent = {}
    for p in result.parents:
        for a in p.attributes:
            attr_to_parent[a.strip()] = p.name

    for g in local_groups:
        g["parent"] = attr_to_parent.get(g["attribute"], g["attribute"])

    save_json(rule_groups, _rules_per_doc_path)
    return {"rule_groups": rule_groups}


def subject_tagging_node(state: MessagesState) -> MessagesState:
    """Размечает, к какому изделию/узлу относится каждая группа правил (subjects.json).

    Нужно, когда документ описывает несколько разных предметов: тогда одинаково названные
    атрибуты означают разное, и применяющий граф не должен смешивать их правила. Шаг
    необязательный — при ошибке файл просто не пишется, и применение работает как раньше.
    Этот же шаг можно запустить отдельно, не перегоняя PDF:
        venv2/Scripts/python -m graphs.structured_documents.subject_tagging
    """
    rule_groups = state.get("rule_groups") or []
    subjects = tag_subjects(rule_groups, clustering_llm)
    if subjects:
        save_json(subjects, _doc_dir / "subjects.json")
    return {"rule_groups": rule_groups}


def rule_cards_node(state: MessagesState) -> MessagesState:
    """Составляет карточки правил (rule_cards.json): документ толкуется один раз при загрузке.

    Рантайм берёт из них принятые формы («SMLS» → «Seamless») и заменяет их без модели. Шаг
    необязательный — при ошибке файл не пишется. Отдельно, без перегонки PDF:
        venv2/Scripts/python -m graphs.structured_documents.rule_cards --doc <папка документа>
    """
    rule_groups = state.get("rule_groups") or []
    try:
        cards_llm = ChatOpenAI(model=settings.llmodel, max_tokens=16000, temperature=0,
                               model_kwargs={"response_format": {"type": "json_object"}})
        save_json(compile_cards(rule_groups, cards_llm), _doc_dir / "rule_cards.json")
    except Exception as e:
        print("RULE CARDS ERROR (пропускаем):", type(e).__name__, str(e)[:200])
    return {"rule_groups": rule_groups}


def table_roles_node(state: MessagesState) -> MessagesState:
    """Роли перечней в тексте групп (table_roles.json): реестр кодов, синонимы, пересчёт, примеры.

    Рантайм строит из них закрытые реестры и таблицы синонимов, вместо того чтобы разбирать
    строки правил регулярками. Шаг необязательный — при ошибке файл не пишется. Отдельно:
        venv2/Scripts/python -m graphs.structured_documents.table_roles --doc <папка документа>
    """
    rule_groups = state.get("rule_groups") or []
    try:
        roles_llm = ChatOpenAI(model=settings.llmodel, max_tokens=16000, temperature=0, timeout=300,
                               model_kwargs={"response_format": {"type": "json_object"}})
        save_json(classify_lists_voted(rule_groups, roles_llm), _doc_dir / "table_roles.json")
    except Exception as e:
        print("TABLE ROLES ERROR (пропускаем):", type(e).__name__, str(e)[:200])
    return {"rule_groups": rule_groups}


builder = StateGraph(MessagesState)
builder.add_node(first_node)
builder.add_node(formatting_text_node)
builder.add_node(grouping_rules_node)
builder.add_node(cluster_groups_node)
builder.add_node(subject_tagging_node)
builder.add_node(rule_cards_node)
builder.add_node(table_roles_node)
builder.add_edge(START, "first_node")
builder.add_edge("first_node", "formatting_text_node")
builder.add_edge("formatting_text_node", "grouping_rules_node")
builder.add_edge("grouping_rules_node", "cluster_groups_node")
builder.add_edge("cluster_groups_node", "subject_tagging_node")
builder.add_edge("subject_tagging_node", "rule_cards_node")
builder.add_edge("rule_cards_node", "table_roles_node")
builder.add_edge("table_roles_node", END)
graph = builder.compile()

inv = graph.invoke({"messages": []})

# for message in inv["messages"]:
#     message.pretty_print()

print(inv.get("rule_groups"))