import json
import re

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
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langgraph.graph import END, START, StateGraph, MessagesState
from PIL import Image as PILImage
import io
import pytesseract
from pydantic import BaseModel, Field
from pathlib import Path


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


vlm = ChatOpenAI(
    model="cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit",
    base_url="https://skillet-appealing-broken.ngrok-free.dev/v1",
    api_key="not-needed",
    max_tokens=4900,
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


def first_node(state: MessagesState) -> MessagesState:
    cached = load_json(_pages_cache)
    if cached is not None:
        print(f"Загружено {len(cached)} страниц из кэша {_pages_cache}")
        return {"messages": [], "pages_data": cached}

    _document = fitz.open(settings.input_path_document)

    received_text = []
    pages_data = []  # <-- сюда будем собирать данные для JSON

    for page_number in range(len(_document)):

        first_page = _document[page_number]

        pix = first_page.get_pixmap(dpi=160)

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
            print(f"OSD detected rotation: {detected_angle}°, confidence: {confidence}")
        except pytesseract.TesseractError as e:
            print(f"OSD не смог определить ориентацию: {e}")

        if detected_angle != 0 and confidence >= 1.0:
            img_rotated = img.rotate(-detected_angle, expand=True)
            print(f"Изображение повёрнуто на {-detected_angle}°")
        else:
            img_rotated = img
            print("Поворот не применялся (угол 0 или низкая уверенность)")

        # img_rotated.save(f"rotated_page_{page_number}_1.png")

        buf = io.BytesIO()
        img_rotated.save(buf, format="PNG")

        image_bytes = pix.tobytes()
        image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        message = HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
            {"type": "text", "text": "Extract text from this page."}
        ])

        response = vlm.invoke([SystemMessage(content=settings.VL_prompt), message])
        token_usage = response.response_metadata.get("token_usage")
        print(token_usage)

        received_text.append(response)

        # собираем запись для JSON
        pages_data.append({
            "page_number": page_number,
            "source_page_index": page_number,
            "detected_angle": detected_angle,
            "confidence": confidence,
            "content": response.content,
            "token_usage": token_usage,
        })

    save_json(pages_data, _pages_cache)

    return {"messages": received_text, "pages_data": pages_data}

def formatting_text_node(state: MessagesState) -> MessagesState:
    cached = load_json(_check_cache)
    if cached is not None:
        print(f"Загружено {len(cached)} страниц после форматирования из кэша {_check_cache}")
        return {"messages": [], "data_after_check": cached}

    llm = ChatOpenAI(model=settings.llmodel, model_kwargs={"response_format": {"type": "json_object"}})

    pages_after_llm = []
    pages = state.get("pages_data")

    if len(pages) == 1:
        # отдельная обработка единственной страницы без пары
        page_copy = dict(pages[0])
        try:
            page_copy["content"] = json.loads(pages[0]["content"])
        except json.JSONDecodeError:
            print("error pass")

        result = llm.invoke([
            SystemMessage(content=settings.llm_prompt),
            HumanMessage(content=json.dumps([page_copy], ensure_ascii=False)),
        ])
        parsed = json.loads(result.content)
        print(f"\n\nsource: {parsed}\n\n")
        result_page = parsed["pages"][0]
        pages_after_llm.append(result_page)
        print(f"\n\n{result.response_metadata.get('token_usage')}\n\npage {result_page}\n\n")

    else:
        for i in range(0, len(pages) - 1):
            page_copy = [dict(pages[i]), dict(pages[i + 1])]
            for index, original in enumerate((pages[i], pages[i + 1])):
                try:
                    page_copy[index]["content"] = json.loads(original["content"])
                except json.JSONDecodeError:
                    print("error pass")

            result = llm.invoke([
                SystemMessage(content=settings.llm_prompt),
                HumanMessage(content=json.dumps(page_copy, ensure_ascii=False)),
            ])
            # ожидаем, что модель вернёт только одну страницу (первую из пары)
            parsed = json.loads(result.content)
            print(f"\n\nsource: {parsed}\n\n")
            result_page = parsed["pages"][0]
            pages_after_llm.append(result_page)
            print(f"\n\n{result.response_metadata.get('token_usage')}\n\npage {result_page}\n\n")

        # последняя страница никогда не была "первой" в паре — обработать отдельно
        last_page = dict(pages[-1])
        try:
            last_page["content"] = json.loads(pages[-1]["content"])
        except json.JSONDecodeError:
            print("error pass")

        result = llm.invoke([
            SystemMessage(content=settings.llm_prompt),
            HumanMessage(content=json.dumps([last_page], ensure_ascii=False)),
        ])
        parsed = json.loads(result.content)
        print(f"\n\nsource: {parsed}\n\n")
        result_page = parsed["pages"][0]
        pages_after_llm.append(result_page)
        print(f"\n\n{result.response_metadata.get('token_usage')}\n\npage {result_page}\n\n")

    # print(f"\n\ndata_after_check {pages_after_llm}\n\n")

    save_json(pages_after_llm, _check_cache)

    return {
        "messages": [AIMessage(content=json.dumps(pages_after_llm, ensure_ascii=False))],
        "data_after_check": pages_after_llm,
    }




grouping_llm = ChatOpenAI(
    model=settings.llmodel,
    max_tokens=10000,
    temperature=0,
)

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

    if len(source_data) == 1:
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


def distill_groups(rule_groups: list[dict]) -> list[dict]:
    """Сжимает тела групп: убирает разъяснительный текст и список исключений.

    Убирает: NOTE —, кросс-референсы между стандартами, а из global — список «reproduced
    exactly» (он уже обработан детерминированно в excluded_attributes, в контекст LLM не нужен).
    """
    distilled = []
    for g in rule_groups:
        kept = []
        for r in g.get("rules", []):
            rl = r.lower().strip()
            if rl.startswith("note"):
                continue
            if rl.startswith("unlike xyz std"):
                continue
            if g.get("type") == "global" and "reproduced exactly" in rl and re.search(r"[•–\n]", r):
                continue
            kept.append(r)
        ng = dict(g)
        ng["rules"] = kept
        distilled.append(ng)
    return distilled


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


builder = StateGraph(MessagesState)
builder.add_node(first_node)
builder.add_node(formatting_text_node)
builder.add_node(grouping_rules_node)
builder.add_node(cluster_groups_node)
builder.add_edge(START, "first_node")
builder.add_edge("first_node", "formatting_text_node")
builder.add_edge("formatting_text_node", "grouping_rules_node")
builder.add_edge("grouping_rules_node", "cluster_groups_node")
builder.add_edge("cluster_groups_node", END)
graph = builder.compile()

inv = graph.invoke({"messages": []})

# for message in inv["messages"]:
#     message.pretty_print()

print(inv.get("rule_groups"))