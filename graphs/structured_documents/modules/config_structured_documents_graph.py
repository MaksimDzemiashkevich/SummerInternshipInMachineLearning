from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path

# Какой PDF экстрактить, можно задать не правкой файла, а переменной окружения:
#   STRUCTURED_DOC_PATH=graphs/structured_documents/data/input/<файл>.pdf
_DEFAULT_DOCUMENT = "graphs/structured_documents/data/input/OGS-STD-001_Standardization_Rules.pdf"


@dataclass
class Settings:
    input_path_document: Path = field(
        default_factory=lambda: Path(os.getenv("STRUCTURED_DOC_PATH", _DEFAULT_DOCUMENT)))
    RULES_STORAGE_PATH = Path("graphs/structured_documents/data/output/rule_groups.json")
    PAGES_DATA_CACHE_PATH: Path = Path("graphs/structured_documents/data/output/pages_data.json")
    DATA_AFTER_CHECK_CACHE_PATH: Path = Path("graphs/structured_documents/data/output/data_after_check.json")
    PROCESSED_DOCS_PATH: Path = Path("graphs/structured_documents/data/output/processed_documents.json")
    VL_model : str = "Qwen/Qwen3-VL-8B-Instruct"
    VL_url : str = "https://skillet-appealing-broken.ngrok-free.dev/v1"
    VL_api_key : str = "not-needed"
    VL_max_tokens : int = 4900
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

    The same holds for TABLES. A table at the top of the page that has no title of its
    own (often only a repeated header row, or no header at all) is the continuation of a
    table from the previous page. Transcribe ALL of its rows as a separate table block
    with an empty title ("title": ""). Never skip it, and never give it the title of the
    next table on the page: that title belongs only to the rows printed below it. A
    caption printed directly above a table (its number and name) is that table's
    "title": put it into the title field of the same table block, not into a separate
    heading block.

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

    BLOCK_GROUPING_PROMPT: str = """Ты раскладываешь пронумерованные блоки нормативного документа (правила нормализации значений атрибутов) по группам правил.

На вход — список блоков {"id", "section", "text"}. Текст блоков НЕ переписывай: ты только назначаешь номера блоков группам. Текст группы соберётся из блоков дословно.

=== ВИДЫ ГРУПП ===
- type="global": положения, действующие на все атрибуты сразу (область применения, общие принципы, «меняется только представление», список исключённых из переформатирования, общие указания для неясных случаев). Все такие блоки — в ОДНУ группу attribute="Global".
- type="local", kind="presentation": правило о том, КАК записывать значение одного атрибута/величины (единица, число знаков, порядок слов, разделители, коды, шаблон формы вида «<n.n> <единица>»). Одна группа = одна величина/атрибут. Название группы — сама величина так, как её называет документ («<величина A>», «<обозначение B>»), а не название раздела.
- type="local", kind="identity": правила о том, какие атрибуты определяют идентичность изделия и когда записи можно считать одним изделием (матрица ключевых атрибутов). Такие блоки НИКОГДА не кладутся в presentation-группы, даже если называют те же атрибуты.
- type="local", kind="class": только если у класса/вида изделия есть СОБСТВЕННЫЙ раздел документа (заголовок раздела — название класса, внутри — его определение, перечень или таблица его атрибутов с формой записи каждого). В такую группу — пункты этого раздела и его таблица атрибутов. Одна группа = один класс; attribute — название класса так, как его называет документ («<класс X>»), без номера раздела. Такие блоки НЕ кладутся в Global: Global — только то, что действует для любого класса. Чего в группу класса НЕ класть: строки матрицы идентичности (они всегда kind="identity"), worked examples и образцы датащитов (они — в группы величин, форму которых показывают), правила о величине, общие для нескольких классов. Если отдельных разделов на класс в документе нет — групп kind="class" не создавай вовсе.

=== НАЗНАЧЕНИЕ БЛОКОВ ===
- Блок может попасть в НЕСКОЛЬКО групп, если говорит о нескольких атрибутах: «<A> and <B> shall…» — в обе группы; строка worked example или образца датащита («Raw record: <A> …; <C> …») — в группу КАЖДОГО атрибута, форму которого она показывает; строка приложения-словаря («<A> | <форма>») — в группу своей величины.
- Пункт, который отсылает к примерам, таблице или приложению (к примерам ниже, к форме в приложении, к кодам таблицы), и сами эти примеры, строки таблицы и строки приложения — в ОДНУ группу, даже если они стоят на другой странице или таблица продолжается без заголовка.
- Раздел «edge cases / implementation guidance»: пункт про конкретный атрибут — в группу этого атрибута (может быть и отдельной группой, если атрибут больше нигде не описан); общий — в Global.
- Таблица, зависящая от класса оборудования (знаки после запятой по классам), — в группу величины, к которой она относится.
- Ссылки на пункты других документов оставь в группе той величины, о которой пункт.
- Каждый id обязан оказаться либо хотя бы в одной группе, либо в skipped. В skipped — только то, в чём нет ни требования, ни примера: титул, оглавление, дисклеймеры, история редакций, предисловие без требований.

=== АЛИАСЫ ===
Для local-групп дай 5–12 коротких терминов: синонимы величины И подписи колонки, под которыми это значение встречается в датащитах РАЗНЫХ классов оборудования документа. Обязательно включи названия атрибутов, которые сам документ использует в примерах, worked examples, образцах датащитов и таблицах для этой величины.

Верни ТОЛЬКО валидный JSON без markdown:
{"groups": [{"attribute": "<величина A>", "type": "local", "kind": "presentation", "aliases": ["<синоним 1>", "<подпись колонки 2>"], "units": [12, 13, 40]}, {"attribute": "Global", "type": "global", "kind": "presentation", "aliases": [], "units": [3, 4]}], "skipped": [0, 1]}"""

    SUBJECT_PROMPT: str = """Ты определяешь, ограничено ли каждое правило документа конкретным изделием (классом оборудования).

На вход — начало документа (область применения) и группы правил {"id", "attribute", "rules"}.

Зачем: если документ даёт для ОДНОГО И ТОГО ЖЕ атрибута разные правила для разных изделий (например «Size» задвижки и «Size» насоса), применяющий граф должен выбрать правило по изделию строки. Если же правило одно на все изделия, привязка к изделию вредна: правило перестанет находиться для остальных изделий.

Ставь предмет группе, ТОЛЬКО если сам текст её правил ограничивает себя изделием: правило сформулировано «для задвижек…», «for <изделие> only…», стоит в разделе, названном по изделию, или документ прямо говорит, что правило действует лишь для этого изделия.

НЕ делают группу предметной:
- примеры, worked examples, образцы датащитов, строки таблиц, где упомянуто изделие (пример всегда про какое-то изделие);
- перечисление в правиле нескольких изделий, к которым оно одинаково применяется;
- таблица, по-разному задающая параметр для разных классов (это одно правило на все классы — оставь без предмета);
- то, что величина чаще встречается у одного изделия;
- исключение для одного изделия внутри группы с общим правилом («…except that for <изделие>…»): общее правило действует для остальных изделий, группа остаётся без предмета.
Сомневаешься — не ставь.

Название предмета — разновидность изделия так, как её называет документ, в нижнем регистре. Если правило честно ограничено несколькими разновидностями, перечисли их через «/» («труба/фитинг»). Группа указывается своим номером id из входных данных (названия групп могут повторяться, номера — нет).

Верни ТОЛЬКО валидный JSON без markdown и пояснений — только группы с предметом, остальные просто не упоминай:

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
    # Карточки правил (rule_cards.py): документ толкуется один раз при загрузке, а не заново
    # на каждой строке. Промпты не должны содержать ничего от конкретного документа.
    RULE_CARD_PROMPT: str = """Ты составляешь «карточки правил» нормативного документа о записи значений атрибутов. Карточка — это те же правила документа, пересказанные однозначно: тот, кто нормализует значение, должен применить её, не толкуя прозу заново.

На вход: общие положения документа (GLOBAL) и несколько групп правил {"id", "attribute", "rules"}. Для КАЖДОЙ группы верни карточку ТОЛЬКО про атрибут этой группы. Если текст группы говорит и о других атрибутах, их не пересказывай.

Поля карточки:
- "id": id группы.
- "written_form": одно предложение — в каком виде значение записывается (порядок частей, пробелы, регистр, разделители, единица).
- "unit": единица, в которой документ велит хранить величину, или "".
- "decimals": сколько знаков после запятой велит документ, или null, если документ не говорит.
- "values": допустимые формы записи, если документ перечисляет набор значений; иначе [].
- "closed": true, если значение вне этого набора не допускается (его нельзя подобрать «по смыслу»); false, если набор — лишь примеры.
- "not_listed": что делать со значением вне набора — так, как говорит документ ("keep and flag", "report", "apply the general rule" …), или "".
- "accepted": пары {"form", "written"} — каждая форма, которую документ признаёт другим написанием значения («X ('a' and 'b' accepted)», «X; we also accept 'a'», строки таблиц «каноническое значение ← принятые формы»). "written" — форма, которой документ велит записывать это значение. Принятая форма — это вход, её заменяют на "written"; оставлять её нельзя. Бери формы ДОСЛОВНО из текста, ничего не придумывай и не достраивай. И "form", и "written" — буквальные написания значения ЦЕЛИКОМ, как они стоят в ячейке («SMLS», «Seamless»), а не описания преобразований («the number only», «TRIM joined to the number», «a hash sign in front of the number») — такие случаи идут в "steps", а не в "accepted".
- "steps": правила записи этого атрибута по порядку применения, каждое — одно повелительное предложение на английском, со ссылкой на номер пункта, если он есть. Раскрывай неоднозначности так, как их решает сам документ (его определения, примеры, общие положения), а не по привычке.
- "exceptions": когда значение оставляется как записано или помечается; [] если нет.
- "unclear": что документ оставляет открытым для этого атрибута (два противоречащих правила, нет правила для частого случая); [] если всё ясно.
- "examples": все примеры преобразования значения этого атрибута, какие есть в тексте группы: пары {"from", "to"} — запись до и после («EXAMPLE: "Cr18-Ni10-Mo" → "18Cr-10Ni-Mo"», строки «X -> Y», пары raw/normalized record или value, образец датащита «было/стало»). Обе стороны ДОСЛОВНО, как в тексте, без кавычек вокруг; если в строке записи несколько атрибутов — только значение этого атрибута. Нет примеров — [].
- "unit_spacing": "space", если документ требует пробел между числом и единицей для этого атрибута (словами или примером), "none", если требует писать слитно, "" если не говорит.

Не выдумывай требований, которых нет в тексте. Значения, формы и единицы — дословно из документа.

Верни ТОЛЬКО валидный JSON без markdown: {"cards": [{"id": 0, "written_form": "...", "unit": "", "decimals": null, "values": [], "closed": false, "not_listed": "", "accepted": [], "steps": [], "exceptions": [], "unclear": [], "examples": [], "unit_spacing": ""}]}"""

    GLOBAL_CARD_PROMPT: str = """Ты пересказываешь общие положения нормативного документа о записи значений атрибутов однозначно — так, чтобы их можно было применять, не толкуя прозу заново.

Верни:
- "steps": общие правила, действующие для любого атрибута, по порядку применения (пробелы, регистр, значения «не применимо»/«уточняется», пересчёт единиц и округление, что делать с неизвестным значением, что никогда не меняется). Каждое — одно повелительное предложение на английском со ссылкой на номер пункта, если он есть. Где документ определяет термин («accepted source form», «canonical value»), пересказывай правило в смысле этого определения.
- "accepted": пары {"form", "written"} для форм, действующих для ЛЮБОГО атрибута (например, написания «не применимо»). Дословно из текста.
- "unclear": что общие положения оставляют открытым; [] если всё ясно.
- "examples": примеры преобразования значений, которые стоят в общих положениях: пары {"from", "to"} дословно, без кавычек вокруг; [] если нет.
- "unit_spacing": "space", если общие положения требуют пробел между числом и единицей, "none", если требуют слитно, "" если не говорят.
- "trailing_zeros": "remove", если общие положения прямо велят убирать нули в конце дробной части (и после округления до заданного числа знаков тоже); "keep", если прямо велят показывать заданное число знаков, дописывая нули (пример «0.63 in» → «0.630 in.»); "" если документ об этом не говорит.

Не выдумывай требований. Верни ТОЛЬКО валидный JSON без markdown: {"steps": [], "accepted": [], "unclear": [], "examples": [], "unit_spacing": "", "trailing_zeros": ""}"""

    # Спецификация атрибута (attribute_spec.py, прототип): одна модель данных на атрибут вместо
    # синонимов, реестров, карточек и шаблонов рендера по отдельности.
    ATTRIBUTE_SPEC_PROMPT: str = """Ты превращаешь правила нормативного документа о записи значений атрибутов в спецификацию, которую исполняет код. Документ может быть написан как угодно: стандарт, служебная записка, словарь данных, чек-лист.

На вход: общие положения документа (GLOBAL) и несколько групп правил {"id", "attribute", "rules"}. Для КАЖДОЙ группы верни одну спецификацию ТОЛЬКО для атрибута этой группы. Учитывай общие положения (десятичный разделитель, регистр, пробел перед единицей) и поправки, которые отменяют прежнее правило: в спецификации — действующая редакция.

Поля:
- "id": id группы.
- "kind": "list" — значение записывается одним из перечисленных документом значений (кодов, наименований); "quantity" — число с единицей (или голое число), которое документ велит пересчитывать, округлять или записывать в определённой единице; "format" — значение собирается по шаблону из частей записи (обозначение, маркировка, буквенно-цифровой класс); "text" — свободный текст, к которому применяется только регистр/пробелы.
- "values" (для list): [{"value": как ЗАПИСЫВАЕТСЯ значение, "accepted": [все входящие написания, которые документ признаёт этим значением]}]. Собери цепочки целиком: если короткое имя по одной таблице означает полное наименование, а полное наименование по другой таблице записывается кодом, то и короткое имя, и полное наименование — в "accepted" этого кода. Написания — дословно из текста.
- "closed" (для list): true, если значение вне списка не подбирается по смыслу (оставляется как есть / передаётся на проверку); false, если список лишь примеры.
- "unit" (для quantity): ОБОЗНАЧЕНИЕ единицы, которое пишется в значении после числа ("mm", "barg", "°C", "m³", "psi", "кг", "МПа"), в том написании, которое велит документ; не название единицы словом и не его грамматическая форма из фразы правила (название единицы в тексте — это её обозначение в значении). "" если документ не задаёт.
- "number_pattern" (для quantity): как записывается число — шаблоном числа ICU/CLDR (тот же синтаксис, что в форматах Excel): "0" — целое; "0.0" — всегда ровно один знак после разделителя, нули дописываются; "0.##" — округлить до двух знаков, нули в конце НЕ писать; "#,##0.00" — с разделителем тысяч; "@@@" — три значащие цифры. В шаблоне только знаки 0 # @ . , — без единицы, текста и условий (условия — в rounding_scope и variants). Точка в шаблоне — ВСЕГДА десятичный разделитель, запятая — ВСЕГДА разделитель тысяч, в каком бы виде документ ни писал числа: десятичную запятую документа задаёт "locale", а не шаблон ("0.00" и "locale": "ru" → «2,50»). Различай «показывать N знаков» ("0" и N нулей) и «округлять до N знаков» / «убирать нули в конце» ("0." и N знаков "#"); общие положения о нулях и знаках действуют и здесь. "" — если документ о записи числа не говорит.
- "locale": код локали CLDR, чьими знаками (десятичный разделитель, разделитель тысяч) документ велит ПИСАТЬ значение — знаки записанного значения, а не те, что он принимает на входе (документ, который принимает «3,53» и велит писать «3.53», — это "en"): "en" — десятичная точка, "ru" — десятичная запятая и пробел между тысячами, "de" — десятичная запятая и точка между тысячами.
- "rounding_scope": "converted", если документ велит округлять только пересчитанные из другой единицы значения, а значение, данное уже в единице записи, не округлять; иначе "all". При "converted" number_pattern описывает запись ПЕРЕСЧИТАННОГО значения; значение в единице записи код сам пишет со всеми его знаками — шаблон ради него не расширяй.
- "rounding": как округлять, если документ это задаёт: "half_up" (по умолчанию, 0.5 вверх), "half_even" (к чётному), "down" (отбросить лишние знаки), "up" (всегда вверх по модулю).
- "unit_space": true, если между числом и единицей пробел; false, если слитно.
- "number_format" (для quantity): "fraction", если документ велит писать число дробью (дюймовые размеры «1/2», «1-1/8» — целая часть через дефис); иначе "decimal".
- "conversions" (для quantity): коэффициенты, которые даёт САМ документ: [{"from": единица, как её пишут на входе, "factor": число, "offset": число}] — значение = (вход × factor) + offset. Только то, что написано в документе; если документ лишь велит пересчитать без коэффициента — []. Единицы, которые документ объявляет той же единицей или другим её написанием, дай парами с "factor": 1 и "offset": 0 — каждое входное написание отдельно.
- "series" (для quantity): если документ велит взять ближайшее значение из фиксированного ряда — этот ряд числами в единице записи; иначе [].
- "instructions": одно-три предложения на английском — как из записи получить значение, если код не справится сам (для format и text — главное поле: порядок частей, разделители, регистр, что убрать, что не добавлять). Пиши так, как сказано в документе, ничего не добавляй.
- "flag_when": когда документ велит оставить значение как есть и передать на проверку (по-английски), или "".
- "variants": если документ пишет значение этого атрибута ПО-РАЗНОМУ для разных изделий (у одного вида изделий — одна единица или формат записи, у другого — другая), — список вариантов: [{"applies_to": для каких изделий, как их называет документ, и все поля выше, которые у этого варианта свои}]. Поля верхнего уровня — то, что общее для всех изделий (или вариант по умолчанию). Если запись одна для всех изделий — [].

Не выдумывай требований. Верни ТОЛЬКО валидный JSON без markdown: {"specs": [{"id": 0, "kind": "list", "values": [], "closed": false, "unit": "", "number_pattern": "", "locale": "en", "rounding": "half_up", "rounding_scope": "all", "unit_space": true, "number_format": "decimal", "conversions": [], "series": [], "instructions": "", "flag_when": "", "variants": []}]}"""

    # Независимая проверка спецификации (spec_verify.py): другая модель сверяет паспорт с правилами.
    SPEC_VERIFY_PROMPT: str = """Ты независимо проверяешь спецификацию атрибута, которую по правилам нормативного документа составила другая модель. Код будет исполнять эту спецификацию буквально, поэтому ошибка в ней станет ошибкой в каждой строке.

На вход: правила документа для атрибута (rules), общие положения документа (general_rules) и спецификация (spec). Сверь каждое поле спецификации с правилами и найди РАСХОЖДЕНИЯ: что пишется, а что лишь принимается на входе, перепутано; в поле единицы не обозначение; число знаков или условие, при котором оно применяется, не те; пропущены принятые формы, значения или исключения; правило для отдельного вида изделий или вида значений не вынесено в вариант; инструкция противоречит правилам или что-то в них упускает; коэффициент пересчёта не тот; шаблон числа (number_pattern), область округления (rounding_scope) или локаль не те, что велят правила и общие положения (о нулях в конце, числе знаков, разделителях). number_pattern — шаблон ICU: точка в нём всегда десятичный разделитель, запятая всегда разделитель тысяч; десятичная запятая документа задаётся полем locale ("ru"), а не запятой в шаблоне. locale — знаки ЗАПИСАННОГО значения: документ, который лишь принимает десятичную запятую на входе, а пишет точку, — это "en", не расхождение. При rounding_scope = "converted" шаблон относится только к пересчитанным значениям: значение, уже данное в единице записи, код не округляет и пишет со всеми его знаками, поэтому шаблон ради таких значений расширять НЕ нужно — это не расхождение.

Указывай только расхождения, которые подтверждает ДОСЛОВНАЯ фраза из rules или general_rules (поле "quote" — скопируй её точно). Для каждого расхождения дай исправленное значение поля целиком, в той же схеме, что в spec. Для поля внутри варианта укажи "variant" — его applies_to; для поля верхнего уровня "variant": "". Не выдумывай требований, которых нет в тексте. Если спецификация верна — пустой список.

Верни ТОЛЬКО валидный JSON без markdown: {"issues": [{"field": "...", "variant": "", "problem": "...", "quote": "...", "value": ...}]}"""

    # Узкая проверка шаблона числа (spec_verify.py): «показывать N знаков» против «округлять до N знаков».
    ZEROS_CHECK_PROMPT: str = """Ты проверяешь одну вещь в спецификации атрибута, составленной по правилам нормативного документа: как записываются знаки после десятичного разделителя.

На вход: правила документа для атрибута (rules), общие положения документа (general_rules), единица записи (unit) и число знаков после разделителя, которое задаёт спецификация (decimals).

Определи по тексту, что велит документ:
- "show" — ВСЕГДА показывать ровно столько знаков, дописывая нули в конце (2.5 → «2.50» при двух знаках);
- "remove" — округлять не более чем до этого числа знаков, а нули в конце дробной части не писать (2.50 → «2.5», 7.0 → «7»).
Правило о нулях в конце из общих положений действует для каждого атрибута, если правило самого атрибута прямо не велит иное. Если документ о нулях в конце не говорит ни в общих положениях, ни в правилах атрибута — "unknown".

В "quote" скопируй дословно фразу из rules или general_rules, на которой основан ответ.

Верни ТОЛЬКО валидный JSON без markdown: {"zeros": "show" | "remove" | "unknown", "quote": "..."}"""

    # Спецификация со свободными полями (attribute_spec_free.py, эксперимент): структуру выбирает модель.
    FREE_SPEC_PROMPT: str = """Ты превращаешь правила нормативного документа о записи значений атрибутов в описание, по которому другой исполнитель сможет однозначно записать любое значение атрибута, не перечитывая документ.

На вход: общие положения документа (GLOBAL) и несколько групп правил {"id", "attribute", "rules"}. Для КАЖДОЙ группы составь описание ТОЛЬКО для атрибута этой группы, с учётом общих положений и поправок (действует последняя редакция). Структуру описания — какие поля, как их назвать, что в них положить — выбери сам так, как лучше всего подходит именно этому атрибуту: форма записи, допустимые значения и их принятые формы, единицы и пересчёт, округление, исключения, различия по видам изделий или видам значений, что делать со значением, к которому правила не подходят. Значения, формы и коэффициенты — дословно из документа; ничего не выдумывай.

Верни ТОЛЬКО валидный JSON без markdown: {"specs": [{"id": 0, "spec": {...}}]}"""

    SPEC_REPAIR_PROMPT: str = """Ты исправляешь спецификацию атрибута, составленную по правилам нормативного документа. Код исполнил её на примерах «было → стало» из самого документа, и часть примеров не сошлась: ниже — что ожидалось и что получилось.

Найди в спецификации поле, из-за которого пример не сходится (не тот kind, лишние или недостающие знаки, единица, не та форма в списке, пропущенная принятая форма, вариант для другого изделия, неточная инструкция), и исправь его так, как требуют ПРАВИЛА документа. Не подгоняй спецификацию под пример, если пример противоречит правилам; не выдумывай требований. Значения и принятые формы — дословно из текста.

Поля спецификации: kind, values, closed, unit, number_pattern (шаблон числа ICU: "0.0" — ровно один знак, "0.#" — до одного знака без нулей в конце; точка в шаблоне — всегда десятичный разделитель, десятичную запятую задаёт locale), locale (знаки разделителей: "en", "ru", "de"), rounding, rounding_scope ("all" | "converted" — округлять только пересчитанные), unit_space, number_format ("decimal" | "fraction" — число дробью, «1-1/8»), conversions, series, instructions, flag_when, variants. Поле, которого нет во входной спецификации, можно добавить.

Верни ТОЛЬКО валидный JSON — спецификацию целиком в той же схеме, что на входе (без поля rules)."""

    # Роли таблиц (table_roles.py): какие перечни в тексте группы правил — списки значений, а
    # какие — нет. Вместо разбора строк таблиц регулярками на рантайме.
    TABLE_ROLES_PROMPT: str = """Ты находишь в тексте групп правил нормативного документа перечни соответствий значений и определяешь, что каждый из них значит. Перечень может быть таблицей в любом виде: строками «колонка: 'значение'; колонка: 'значение'», колонками через «|», строками «A -> B», списком в прозе.

На вход: общие положения документа (GLOBAL) — только как контекст, и несколько групп {"id", "attribute", "rules"}. Для КАЖДОЙ группы верни все перечни в её тексте, каждый с ролью:
- "register": таблица кодов — перечень «описание значения → код (сокращение)», и правило велит записывать в ячейку КОД из этой таблицы вместо описания. Код подставляется только для описания из таблицы; описанию, которого в таблице нет, код не подбирают. Пары: "form" — описание значения, "value" — код. Если "value" — полная запись того же значения, а не код, это "synonyms". Пересчёт одной системы обозначений в другую (номинальный размер одной системы → другой, номинальное давление → класс, число → то же число с обозначением) — тоже "synonyms".
- "synonyms": перечень других написаний значений — входящее написание из перечня заменяется на основную запись, а значение, которого в перечне нет, обрабатывается остальными правилами (перечень не обязан быть полным). Сюда же — перечни пересчёта одного обозначения в другое (одна система обозначений → другая). Пары: "form" — принятое (входящее) написание, "value" — основная запись. Если в ячейке перечислено несколько написаний (через «;», «,», «or»), дай каждое отдельной парой. Основная запись — тоже форма самой себя, её отдельной парой не давай. Одна и та же форма, ведущая к разным записям в зависимости от условий, — не синоним, её не давай.
- "conversion": коэффициенты или формулы пересчёта единиц.
- "examples": примеры преобразования «было → стало» (в том числе строки тестовых записей со ссылками на номера правил).
- "other": всё прочее — перечни допустимых значений без кодов, указатели атрибутов, перечни атрибутов изделия, единицы и знаки после запятой, номера правил, описания полей. И обязательно: другие названия САМОГО атрибута или изделия (синонимы имён атрибутов, наименований изделий, заголовков колонок) и расшифровки сокращений — это не значения, а имена, их пары не давай.
Пары давай только для "register" и "synonyms"; для остальных ролей "pairs": [].

Направление важно: "value" — это то, что ПИШЕТСЯ в результат по документу; "form" — то, что может прийти на вход. Определи это по заголовкам колонок и по тексту правила вокруг перечня, а не по порядку колонок.
"form" и "value" — ДОСЛОВНО из текста, каждое — написание значения целиком (без кавычек вокруг), а не описание преобразования («the number only», «joined to the number»). Ничего не придумывай и не достраивай. Номер правила, номер пункта или ссылка («R-01», «see 7.2») — не значение.
"title" — заголовок перечня, как в тексте, или "".
"whole_attribute": для "register" — true, если по документу КАЖДОЕ значение этого атрибута записывается кодом из этой таблицы (правило атрибута и есть «код из таблицы»); false, если таблица задаёт коды лишь для части значений, а остальные значения атрибута пишутся по формату из правила (число, обозначение, буква). Для остальных ролей false.
"attribute": имя атрибута, ЗНАЧЕНИЯ которого перечислены, как его называет документ. Текст группы может содержать перечни других атрибутов (соседний раздел, общая таблица) — тогда здесь их атрибут, а не атрибут группы. Если перечень про значения атрибута группы — имя группы.
"any_attribute": true, только если документ применяет перечень к значению ЛЮБОГО атрибута (например, написания «не применимо», «уточняется»); false, если перечень — про значения одного атрибута или вообще не про значения атрибутов (наименования изделий, имена атрибутов, единицы).

Верни ТОЛЬКО валидный JSON без markdown: {"groups": [{"id": 0, "lists": [{"title": "", "attribute": "", "role": "register", "whole_attribute": true, "any_attribute": false, "pairs": [{"form": "...", "value": "..."}]}]}]}. У группы без перечней "lists": []."""
