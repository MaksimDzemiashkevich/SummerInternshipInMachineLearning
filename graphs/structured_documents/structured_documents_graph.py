import json

from graphs.structured_documents.modules.config_structured_documents_graph import Settings
from dotenv import load_dotenv

load_dotenv()
settings = Settings()

import base64
import fitz
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langgraph.graph import END, START, StateGraph, MessagesState
from PIL import Image as PILImage
import io
import pytesseract
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from pathlib import Path


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
_model = SentenceTransformer('all-MiniLM-L6-v2')


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
    attribute: str = Field(description="Name of the attribute this group of rules belongs to, determined from the meaning of the text")
    rules: list[str] = Field(description="List of rules/requirements relating to this attribute")

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
    cached = load_json(settings.PAGES_DATA_CACHE_PATH)
    if cached is not None:
        print(f"Загружено {len(cached)} страниц из кэша {settings.PAGES_DATA_CACHE_PATH}")
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

    save_json(pages_data, settings.PAGES_DATA_CACHE_PATH)

    return {"messages": received_text, "pages_data": pages_data}

def formatting_text_node(state: MessagesState) -> MessagesState:
    cached = load_json(settings.DATA_AFTER_CHECK_CACHE_PATH)
    if cached is not None:
        print(f"Загружено {len(cached)} страниц после форматирования из кэша {settings.DATA_AFTER_CHECK_CACHE_PATH}")
        return {"messages": [], "data_after_check": cached}

    llm = ChatOpenAI(model=settings.llmodel, temperature=0, model_kwargs={"response_format": {"type": "json_object"}})

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

    save_json(pages_after_llm, settings.DATA_AFTER_CHECK_CACHE_PATH)

    return {
        "messages": [AIMessage(content=json.dumps(pages_after_llm, ensure_ascii=False))],
        "data_after_check": pages_after_llm,
    }







# --- локальная VL-модель, используется здесь БЕЗ картинки — только текст ---
grouping_llm = ChatOpenAI(
    model=settings.VL_model,
    base_url=settings.VL_url,
    api_key=settings.VL_api_key,
    max_tokens=10000,  # с запасом: таблицы/длинные страницы могут дать длинный ответ
    temperature=0,
)

# добавляем к промпту явное требование формата — раз мы больше не полагаемся
# на with_structured_output (guided decoding), модель должна получить точную
# инструкцию по схеме прямо текстом

# JSON_SCHEMA_INSTRUCTION = """
#
# === ФОРМАТ ОТВЕТА (СТРОГО ОБЯЗАТЕЛЬНО) ===
#
# Верни ТОЛЬКО валидный JSON, без markdown-разметки (без ```json), без пояснений
# до или после — ничего, кроме самого JSON. Строго такая структура:
#
# {
#   "groups": [
#     {
#       "attribute": "название атрибута",
#       "rules": ["правило 1", "правило 2"]
#     }
#   ]
# }
#
# Если ни одного правила не найдено — верни {"groups": []}."""

def grouping_rules_node(state: MessagesState) -> MessagesState:
    source_data = state.get("data_after_check") or state.get("pages_data")

    # Если группы правил уже сохранены на диск — берём их и не гоняем LLM повторно.
    cached = load_json(settings.RULES_STORAGE_PATH)
    if cached is not None:
        print(f"Загружено {len(cached)} групп правил из кэша {settings.RULES_STORAGE_PATH}")
        return {"rule_groups": cached}

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
                # "\n\nThe following attributes have already been identified: "
                # + ", ".join(known_attributes)
                # + ". If the new rule relates to one of them, use EXACTLY "
                #   "the same attribute name — do not invent a new wording."
                ""
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
            all_groups.append(g_dict)
            if g_dict["attribute"] not in known_attributes:
                known_attributes.append(g_dict["attribute"])

    if len(source_data) == 1:
        call_llm([prepare(source_data[0])])
    else:
        for i in range(len(source_data)):
            call_llm([prepare(source_data[i])])

    # --- сохраняем результат в файл (он же служит кэшем при следующем запуске) ---
    save_json(all_groups, settings.RULES_STORAGE_PATH)

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