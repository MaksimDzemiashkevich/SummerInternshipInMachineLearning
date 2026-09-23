from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path

# Какой PDF экстрактить, можно задать не правкой файла, а переменной окружения:
#   STRUCTURED_DOC_PATH=graphs/structured_documents/data/input/<файл>.pdf
_DEFAULT_DOCUMENT = "graphs/structured_documents/data/input/FGX-STD-227_fire_gas_lifesafety_equipment_normalization_rules.pdf"


@dataclass
class Settings:
    input_path_document: Path = field(
        default_factory=lambda: Path(os.getenv("STRUCTURED_DOC_PATH", _DEFAULT_DOCUMENT)))
    RULES_STORAGE_PATH = Path("graphs/structured_documents/data/output/rule_groups.json")
    PAGES_DATA_CACHE_PATH: Path = Path("graphs/structured_documents/data/output/pages_data.json")
    DATA_AFTER_CHECK_CACHE_PATH: Path = Path("graphs/structured_documents/data/output/data_after_check.json")
    PROCESSED_DOCS_PATH: Path = Path("graphs/structured_documents/data/output/processed_documents.json")
    VL_model : str = "cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit"
    VL_url : str = "https://skillet-appealing-broken.ngrok-free.dev/v1"
    VL_api_key : str = "not-needed"
    VL_max_tokens : str = 4900
    llmodel : str = "gpt-5.6-luna"
    VL_prompt: str = """You perform precise structured extraction of a document page's content, but first you analyze the image.

    Return ONLY valid JSON, without markdown formatting (no ```json), and without any explanations before or after. Use strictly the following format:

    {
      "blocks": [
        {"type": "...", "...": "..."},
        {"type": "...", "...": "..."},
        {"type": "...", "...": "..."}
      ]
    }

    where "type" can be paragraph, heading, or table.
    If the type is heading, it must contain the field "level".

    The "level" field of a heading is an INTEGER representing the heading hierarchy depth
    (1 = top-level heading, 2 = subsection, 3 = sub-subsection, etc.),
    NOT the document section number.

    {"type": "heading", "level": 1, "text": "..."}

    A table contains a title, a list of columns, and rows represented as an array.

    {"type": "table", "title": "...", "columns": [...], "rows": [{...}]}

    Regular text is represented as paragraphs.

    {"type": "paragraph", "text": "..."}

    Blocks must be returned in the document reading order.

    BLOCK ORDER RULE (CRITICALLY IMPORTANT):

    The blocks in the output array "blocks" must appear strictly in the order in which
    the content is physically located on the page from top to bottom—the same way a
    person would read the page by moving their eyes downward.

    DO NOT group content by semantic meaning instead of physical order.

    For example, if a page starts with a list (paragraphs/lines without a heading),
    and BELOW it there is a section heading with its own text, then the list must
    appear FIRST in the output, followed by the heading and its text, even if the
    heading appears to be the "main" or "current" logical section.

    Do not move a heading block to the beginning of the output simply because it seems
    structurally important. Base the ordering solely on the vertical position of the
    content in the image: whatever appears higher on the page must appear earlier in
    the "blocks" array.

    Before producing the final JSON, mentally scan the page from top to bottom and
    verify that the order of blocks in your JSON matches the order in which they
    appear on the page. If not, correct the order.

    IMPORTANT ABOUT INCOMPLETE FRAGMENTS AT PAGE BOUNDARIES:

    The text on a page may begin in the middle of a sentence (continued from the
    previous page) or end before the sentence is complete (continued on the next
    page). This is NORMAL and expected.

    IN SUCH CASES, you MUST extract the text exactly as it appears, even if:

    - it starts with a lowercase letter or is clearly a continuation of a sentence;
    - it ends in the middle of a sentence at the bottom of the page.

    Do NOT omit such text under the pretext of "reconstructing" or "inferring"
    missing content.

    You are not inventing anything—you are simply transcribing what is actually
    written in the image, even if it is an incomplete sentence.

    This is different from hallucination:
    hallucination means writing something that is NOT present in the image.
    Transcribing a truncated sentence means copying text that IS present, even if it
    does not form a complete sentence.

    TABLE RULES:

    First analyze the table headers, and only then structure the rows.
    This is necessary to correctly associate cell values with their corresponding
    columns.

    1. If a column has subheaders (for example, the header
       "Seat Surface Typical Specifications Grade" has the subheaders
       "Cast", "Forged", and "Welded"), then name the key using
       "Header/Subheader". For example:

       {
         "Seat Surface Typical Specifications Grade/Cast": "ASTM A217 (CA15)",
         "Seat Surface Typical Specifications Grade/Forged": "ASTM A182 (F6a)",
         "Seat Surface Typical Specifications Grade/Welded": "AWS A5.9 ER410"
       }

    2. If the same column header text appears twice in different parts of the table
       (for example, on the left and on the right as part of a mirrored wide-table
       layout), these are two physically different columns.

       Each has its own subheader.

       Use the combination "Header/Subheader" so that the keys are unique and
       non-conflicting. For example:

       {
         "Class Designation/Nominal Size DN": "25",
         ...,
         "Class Designation/Nominal Pipe Size NPS": "1"
       }

       NEVER create two keys with the same name in the same JSON object.
       In JSON, the second value would overwrite the first, causing data loss.

    3. If a column has no subheader, keep it as a regular key with a string value.

    4. Each table row must be represented as a separate object in the "rows" array.

    5. If a cell is visually merged in the original table (rowspan),
       duplicate its value into EVERY row to which it applies.
       Do not omit rows because of merged cells.

    6. Numbers, codes, and identifiers (for example, "1", "5", "5A")
       must be transcribed exactly as shown in the image,
       even if the sequence appears inconsistent.
       Do not renumber or "correct" them.

    7. If a cell is empty or contains "NA", record it exactly as
       "NA" or "".

    EXAMPLE — table with subheaders (Trim):

    {
      "Trim Number": "6",
      "Nominal Trim": "F6 and Cu-Ni",
      "Seat Surface Typical Specifications Grade/Cast": "ASTM A217 (CA15)",
      "Seat Surface Typical Specifications Grade/Forged": "ASTM A182 (F6a)",
      "Seat Surface Typical Specifications Grade/Welded": "AWS A5.9 ER410",
      "Stem Hardness (HB)": "200 min 275 max"
    }

    EXAMPLE — table with duplicated left/right headers (Wall Thickness):

    {
      "Class Designation/Nominal Size DN": "25",
      "Minimum Wall Thickness tm mm (in.)/150": "6.4 (0.25)",
      "Minimum Wall Thickness tm mm (in.)/300": "6.4 (0.25)",
      "Minimum Wall Thickness tm mm (in.)/600": "7.9 (0.31)",
      "Minimum Wall Thickness tm mm (in.)/900": "12.7 (0.50)",
      "Minimum Wall Thickness tm mm (in.)/1500": "12.7 (0.50)",
      "Minimum Wall Thickness tm mm (in.)/2500": "15.0 (0.59)",
      "Class Designation/Nominal Pipe Size NPS": "1"
    }

    Before producing the final JSON, check every row of every table for duplicate
    top-level keys.

    If duplicates are found, you should have used the
    "Header/Subheader" format. Correct them.

    ROTATION RULES:

    - If a table or text in the image is rotated, rotate it into the correct
      orientation before reading it, and then extract the content as a normal,
      horizontally oriented table or text.

    WHAT TO SKIP (do not include in "blocks"):

    - copyrights;
    - legal disclaimers and liability notices;
    - headers, footers, page numbers, publisher notes
      ("Provided by...", "No reproduction...");
    - table of contents pages (lists of sections with dot leaders and page numbers);
    - diagrams and schematics.

    PROHIBITED:

    - Never reproduce long sequences of repeated characters
      (dot leaders, vertical bars, etc.)—simply omit them.
    - Do not use your own knowledge of what such documents usually contain.
    - Do not add, infer, or reconstruct content that is not present in the image.
    - Do not fill a cell with "NA" by default.
      Use "NA" only if the image actually contains "NA".
    - Do not copy values from one column into another.
      Read every cell independently, even if the values appear similar.
    - If the page contains very little content (for example, only a title page),
      return "blocks" containing exactly what is present and nothing more.
    - If you are unsure whether text exists in the image, do not invent it.

    If the page contains no meaningful content blocks (only metadata that should be
    ignored), return:

    {"blocks": []}

    REMINDER — THE MOST IMPORTANT RULE FOR TABLES:

    Before producing the final JSON, check every table one more time.

    If a column header has a subheader, use the format
    "Header/Subheader" as the key name.

    Never output subheaders as standalone flat keys without their parent header,
    and never allow two identical keys within the same row.
    """
    llm_prompt: str = """You are a business assistant responsible for validating and normalizing data extracted from a document.

    The input is a JSON containing either ONE or TWO pages:

    - If there are TWO pages:
      the first page is the one that must be validated, corrected, and returned.
      The second page is provided ONLY as context to determine whether the last paragraph
      of the first page is incomplete and continues at the beginning of the second page.
      Do NOT return the second page—it is provided solely for validation.

    - If there is only ONE page:
      it is the last page of the document (there is no paired page).
      Process and return only that page.

    Each page contains a field called "content", which stores a JSON STRING
    (not a JSON object, but a string that must be parsed) with the following structure:

    {"blocks": [{"type": "heading"/"table"/"paragraph", ...}]}

    === OUTPUT FORMAT ===

    Return ONLY valid JSON containing a list with EXACTLY ONE page object:
    the corrected first page.

    (Even if the input contains two pages, the output must always contain only one.)

    The returned page must preserve the same "content" field containing the corrected
    JSON string with "blocks".

    Copy all other page fields unchanged:

    - page_number
    - source_page_index
    - detected_angle
    - confidence
    - token_usage

    UNDER NO CIRCUMSTANCES wrap the response in markdown
    (no ```json ... ```).

    Do not add explanations before or after the JSON.
    Return only the JSON itself.

    Your task is to validate the blocks of the returned page on three independent levels
    and return the corrected result.

    === 1. TABLE VALIDATION: column/value correspondence ===

    For every table, verify that each cell value actually belongs to its corresponding
    column (header/subheader).

    Sometimes during extraction, a value may have shifted into an adjacent column.

    Check:

    - Does the value's type or format match what is expected for that column?
      (For example, a numeric value where a number is expected,
      or a unit matching the column header.)

    If you find a clear mismatch, move the value into the correct column.

    If you are uncertain (the mismatch is not obvious),
    DO NOT modify it.

    Only correct clear and well-justified errors.

    === 2. TABLE VALIDATION: symbolic values / variables ===

    Sometimes a table cell contains a symbolic designation
    (letter, abbreviation, variable)
    rather than the actual value.

    Its meaning may be explained elsewhere in the document:

    - in a note below the table;
    - in a paragraph before or after the table;
    - in a footnote.

    If you encounter such a symbol:

    - Search for its explanation in any available blocks
      (paragraphs, headings, or other tables)
      on the current page and, if provided, on the context page.

    - If an unambiguous explanation is found,
      DO NOT replace the symbol inside the table cell.
      The symbol may be part of the document's official notation.
      This is NOT considered an extraction error and does not require correction.

    - If no explanation is found,
      also leave the cell unchanged.
      This is a document feature rather than an extraction error.

    === 3. MERGING BROKEN PARAGRAPHS ===

    There are two independent checks.

    If two pages are provided, apply both.
    If only one page is provided, apply only rule (B).

    --- (A) End of returned page → beginning of context page ---

    If the last paragraph block of the FIRST (returned) page
    ends in the middle of a sentence,
    and the first paragraph block of the SECOND (context) page
    is clearly the continuation of that sentence
    (starts with a lowercase letter and syntactically continues the unfinished sentence),

    append that continuation to the end of the last paragraph
    of the first page.

    Do NOT return the second page.

    Its text should appear only inside the merged paragraph
    of the first page.

    --- (B) Beginning of returned page → possible continuation from previous page ---

    Inspect the FIRST block of the returned page.

    If it clearly looks like the continuation of an unfinished sentence
    (starts with a lowercase letter and is obviously a sentence fragment
    rather than an independent paragraph),

    then its beginning was located on the previous page,
    which has already been processed and is not available now.

    In this case, REMOVE the entire block from the returned page:

    - do not try to reconstruct it;
    - do not leave it unchanged;
    - simply remove it from the "blocks" array.

    This prevents duplicated text,
    because that fragment has already been appended
    to the previous page during the earlier processing step.

    Apply rule (B) ONLY when the continuation is obvious.

    If you are uncertain
    (for example, the paragraph is simply short but grammatically complete),

    DO NOT remove it.

    If the returned page is the first page of the document,
    there normally should not be a continuation at its beginning.
    Only remove the block if there is strong evidence that it is incomplete.

    --- General rule ---

    Merge or remove ONLY genuinely broken sentences.

    If two adjacent paragraphs merely discuss the same topic
    but are both grammatically complete,
    they are separate paragraphs and must remain unchanged.

    If the input contains no extraction errors
    and no broken paragraphs,
    return the data unchanged in exactly the same format
    (a list containing one page object).

    Return the JSON STRICTLY in the following format:

    {
      "pages": [
        {...}
      ]
    }
    using the top-level key "pages", which must contain exactly one page.
    """
    TABLE_EXTRACTION_PROMPT = """Ты извлекаешь правила стандартизации из нормативного документа (стандарт/спецификация) в нефтегазовой отрасли.

На вход — JSON со страницей, где есть блоки: heading, paragraph, table.

=== ГЛАВНОЕ: ГЛОБАЛЬНЫЕ и ЛОКАЛЬНЫЕ правила ===
Для каждого правила определи, к чему оно относится:
- "global" — правило, которое применяется ко МНОГИМ или ВСЕМ атрибутам, либо описывает общий шаблон (например: «все температуры указывать в °C», «все линейные размеры переводить в мм», «keyword + число → слитный код без пробела», «меняется только представление, а не техническая суть»). Такие правила собирай в ОДНУ группу с attribute="Global", type="global".
- Если секция — это СПИСОК атрибутов, которые НЕЛЬЗЯ переформатировать («reproduce exactly as given», список исключений), верни ВЕСЬ список ЦЕЛИКОМ как ОДНО правило типа "global" в одной группе attribute="Global". НЕ создавай отдельные local-группы для пунктов такого списка.
- "local" — правило про ОДИН конкретный атрибут (trim, gate, class, dn, nps, end connection, pressure class, temperature, dimension, hardness, handwheel, bolt thread, shipping position...). Группируй по этому атрибуту, type="local".

=== ГРУППИРОВКА ЛОКАЛЬНЫХ ПРАВИЛ ===
- Каждое правило относи к ТОМУ конкретному атрибуту, о котором оно говорит. Название группы = сам атрибут.
- НЕ создавай «обобщающую» группу, которая накрывает несколько РАЗНЫХ атрибутов. Например, «Class → CL», «Trim → T», «DN → DN», «NPS → NPS» — это ЧЕТЫРЕ разных атрибута, а не одна группа «keyword-number designation».
- НЕ используй названия разделов документа («Units of Measurement», «Scope», «Keyword-Number Designation») как названия групп.
- Каждое предложение с самостоятельным требованием — отдельное правило.
- Если в подсказке перечислены уже существующие группы — при совпадении темы используй ТОЧНО такое же название, не создавай дубль.

=== ТАБЛИЦЫ ===
- Таблица — это набор правил. Каждую строку с непустыми значениями относи к атрибуту, о котором она говорит.
- Если таблица перечисляет сокращения/соответствия для НЕСКОЛЬКИХ атрибутов («Class → CL», «Trim → T», «DN → DN», «NPS → NPS»), разбей её на отдельные группы по атрибутам.
- Если таблица целиком про ОДИН атрибут (например таблица свойств trim), это одна группа.
- Обработай таблицу ПОЛНОСТЬЮ.

=== ПРИМЕР ===
Раздел «Keyword-Number Designation»:
- общее правило «keyword + число → слитный код без пробела» → type="global", attribute="Global";
- «Class → CL; Class 300 → CL300» → type="local", attribute="class";
- «Trim → T; Trim 8A → T8A» → type="local", attribute="trim";
- «DN → DN; DN 80 → DN80» → type="local", attribute="dn";
- «NPS → NPS; NPS 3 → NPS3» → type="local", attribute="nps".

=== ДЕТЕРМИНИРОВАННЫЕ ПРАВИЛА (Keyword → Root) ===
Детерминированное правило — это безусловное соответствие «ключевое слово → код/корень», которое применяется ВО ВСЕХ обстоятельствах и не зависит ни от каких условий. Это не только сокращения (Class → CL), но и аббревиатуры (NPS → NPS, DN → DN) и любые фиксированные коды (Class → CLASSATUM, Trim → TRIMMEUS).
Такое правило запиши отдельной строкой в СТРОГОМ формате:
Keyword: 'Class'; Invented Root: 'CLASSATUM'
Правило, у которого есть условие или зависимость от контекста («если...», «в зависимости от...»), НЕ детерминированное — формулируй его обычным текстом как семантическое правило.

=== ПРИМЕРЫ ===
Строки «EXAMPLE: …» в документе — это поясняющие примеры к конкретному правилу, а НЕ самостоятельные правила. Не выноси их в отдельное правило — оставляй как часть того правила, к которому они относятся.

=== АЛИАСЫ (для local-групп) ===
Для каждой local-группы сгенерируй поле aliases — список терминов/синонимов/аббревиатур, которые типично покрывает эта группа в нефтегазовой терминологии. Пример: для группы «class» — «grade», «rating», «PN», «pressure rating», «class designation». Дай 3–8 коротких терминов (не предложений).

=== ЧТО ПРОПУСКАТЬ ===
- титульные страницы, метаданные, сведения об издателе/правообладателе/датах;
- legal disclaimers, liability notices, copyright;
- оглавление, списки литературы/библиографию;
- контактную информацию, сведения о сертификации.
НЕ пропускай вводные/декларативные разделы (Scope): если там есть «shall/требование», это правило типа "global".

=== ТОЧНОСТЬ ===
- Передавай правила ДОСЛОВНО, не перефразируй, не додумывай, не сокращай числа/единицы/условия.
- Отвечай на языке оригинала документа.

Верни ТОЛЬКО валидный JSON без markdown и пояснений:

{"groups": [{"attribute": "trim", "type": "local", "aliases": ["trim number", "trim code"], "rules": ["..."]}, {"attribute": "Global", "type": "global", "rules": ["..."]}]}"""

    SUBJECT_PROMPT: str = """Ты определяешь, ЧЕМ управляет каждая группа правил: к какому изделию, узлу или классу оборудования она относится.

На вход — выдержка из общих положений документа и список групп правил (название + фрагмент текста).

Зачем: один документ может описывать несколько разных предметов (например, задвижки и насосы), и тогда правила с одинаковым названием атрибута означают разное. Разметка предмета позволяет не смешивать их при применении.

Правила разметки:
- Предмет — это ВЕЩЬ, которой касается правило (изделие, узел, класс оборудования), а не тема атрибута («материал», «температура») и не вид источника («таблица», «пункт»).
- Если весь документ посвящён одному предмету — присвой ОДИН И ТОТ ЖЕ предмет всем группам. Это обычный случай, и это правильный ответ.
- Разные предметы назначай, только если документ действительно разделяет правила по предметам явно.
- Если группа относится КО ВСЕМ предметам документа сразу — НЕ включай её ни в один предмет, просто не упоминай её номер. Не придумывай для таких групп зонтичное название («оборудование», «изделие», «общее»): пустая разметка и означает «применимо ко всему».
- Название предмета — это одна конкретная разновидность изделия так, как её называет документ («задвижка», «труба», «кабель»), в нижнем регистре. Если группа честно относится к нескольким разновидностям, перечисли их через «/» («труба/фитинг»).
- Не используй в названии предмета слова-обобщения без конкретики и не склеивай разные разновидности запятыми в одно название.
- Каждая группа попадает ровно к одному предмету, и указывается своим номером id из входных данных (названия групп могут повторяться, номера — нет).

Верни ТОЛЬКО валидный JSON без markdown и пояснений:

{"subjects": [{"name": "gate valve", "groups": [0, 2, 5]}]}"""

    CLUSTERING_PROMPT: str = """Ты группируешь названия атрибутов в родительские темы.

На вход — список названий атрибутов. Сгруппируй в одну родительскую тему ТОЛЬКО те атрибуты, у которых есть ОБЩЕЕ существительное (одно и то же слово). Примеры:
- «class», «line class», «operating class», «service class» → родитель «class» (общее слово «class»);
- «temperature», «temperature range» → родитель «temperature» (общее слово «temperature»);
- «category», «handling category» → родитель «category».

НЕ группируй по теме/смыслу, если общего слова НЕТ. Например «dn», «nps», «size» — общего существительного нет, оставляй каждый сам себе родителем. Не выдумывай обобщающее имя.

Правила:
- Каждый атрибут относится ровно к одному родителю.
- Родитель = общее существительное (слово из названий).
- Если у атрибута нет общего существительного с другими — он сам себе родитель.

Верни ТОЛЬКО валидный JSON без markdown и пояснений:

{"parents": [{"name": "class", "attributes": ["class", "line class", "operating class"]}]}"""