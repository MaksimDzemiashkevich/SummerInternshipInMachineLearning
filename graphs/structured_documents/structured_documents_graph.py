import json

from graphs.structured_documents.modules.config_structured_documents_graph import Settings
from dotenv import load_dotenv

load_dotenv()
settings = Settings()

# Кэши привязаны к конкретному документу, чтобы несколько PDF можно было
# экстрактить последовательно и накапливать правила в rule_groups.json.
_doc_stem = settings.input_path_document.stem
_out_dir = settings.RULES_STORAGE_PATH.parent
_pages_cache = _out_dir / f"pages_data_{_doc_stem}.json"
_check_cache = _out_dir / f"data_after_check_{_doc_stem}.json"

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
    rules: list[str] = Field(description="List of rules/requirements in this group")

class GroupingResult(BaseModel):
    groups: list[RuleGroup] = Field(description="List of all found rule groups by attribute")

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

def grouping_rules_node(state: MessagesState) -> MessagesState:
    source_data = state.get("data_after_check") or state.get("pages_data")

    # Манифест: какие документы уже добавлены в rule_groups.json.
    processed_docs = load_json(settings.PROCESSED_DOCS_PATH) or []
    if _doc_stem in processed_docs:
        existing = load_json(settings.RULES_STORAGE_PATH) or []
        print(f"Документ {_doc_stem!r} уже обработан — пропускаем группировку ({len(existing)} групп)")
        return {"rule_groups": existing}

    # Стартуем с уже накопленных правил (от предыдущих документов) и дописываем новые.
    all_groups: list[dict] = list(load_json(settings.RULES_STORAGE_PATH) or [])
    known_attributes: list[str] = [
        g["attribute"] for g in all_groups if isinstance(g, dict) and "attribute" in g
    ]

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

    # --- сохраняем накопленный результат и помечаем документ обработанным ---
    save_json(all_groups, settings.RULES_STORAGE_PATH)
    processed_docs.append(_doc_stem)
    save_json(processed_docs, settings.PROCESSED_DOCS_PATH)

    return {"rule_groups": all_groups}

builder = StateGraph(MessagesState)
builder.add_node(first_node)
builder.add_node(formatting_text_node)
builder.add_node(grouping_rules_node)
builder.add_edge(START, "first_node")
builder.add_edge("first_node", "formatting_text_node")
builder.add_edge("formatting_text_node", "grouping_rules_node")
builder.add_edge("grouping_rules_node", END)
graph = builder.compile()

inv = graph.invoke({"messages": []})

# for message in inv["messages"]:
#     message.pretty_print()

print(inv.get("rule_groups"))