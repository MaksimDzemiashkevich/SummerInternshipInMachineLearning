from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Settings:
    input_path_document: Path = Path("data/input/API-600.pdf")
    RULES_STORAGE_PATH = Path("data/output/rule_groups.json")  # можно вынести в settings
    VL_model : str = "cyankiwi/Qwen3-VL-8B-Instruct-AWQ-4bit"
    VL_url : str = "https://skillet-appealing-broken.ngrok-free.dev/v1"
    VL_api_key : str = "not-needed"
    VL_max_tokens : str = 4900
    llmodel : str = "gpt-4o"
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
    grouping_rules_model_prompt: str = """You are a business assistant specializing in the analysis of technical standards and regulatory documents in the oil and gas industry (API standards and similar).

    The input is a JSON containing a list of document pages. Each page has a "content" field containing structured blocks (heading, paragraph, table) extracted from the document.

    Your task is to identify technical rules, requirements, or conditions in the document and group them by the attribute (characteristic, parameter, equipment component, etc.) to which they apply.

    Rules may originate from blocks of ALL types:

    - "paragraph" — textual requirements and normative statements;
    - "table" — specific values (wall thicknesses, pressures, dimensions, materials, etc.) represented by table rows and columns (details below);
    - "heading" — use only as contextual information about the section. Headings themselves are NOT rules and must not be treated as an independent source of rules.

    An "attribute" is the specific characteristic or component that a rule refers to.
    Examples include:

    - body material;
    - minimum wall thickness;
    - pressure class;
    - operating temperature;
    - valve seat hardness.

    Determine the attribute name from the meaning of the text.
    It is not explicitly provided in the document—you must formulate it yourself.

    Keep the attribute concise and clear (2–4 words), using either English or Russian, matching the language of the source document.

    Formulate each rule as closely as possible to the original document without changing its technical meaning.

    Do NOT paraphrase creatively or shorten the wording if doing so would omit exact numbers, units, applicability conditions, or referenced standards.

    If preserving the technical meaning requires a longer formulation, use it.
    Brevity must never come at the expense of precision.

    The same attribute may appear across multiple pages.
    Group all rules belonging to the same attribute together, even if they occur in different parts of the document.

    Ignore text that is not a technical rule or requirement, such as:

    - introductory text;
    - references to other sections without substantive requirements;
    - general descriptions lacking normative conditions.

    Do not include such text in any group.

    === RULES FROM TEXT (PARAGRAPHS AND DEFINITIONS) ===

    Technical requirements and restrictions may appear not only in explicitly numbered clauses
    (such as 4.1 or 5.1.1), but also inside paragraphs within the
    "Terms and Definitions" section.

    For example, a definition may contain a restriction on how the term may be used or interpreted, such as:

    "not a measurable quantity and shall not be used for calculations unless otherwise specified."

    Such statements are also rules and must be extracted separately,
    even if they are embedded within a definition rather than presented as a numbered clause.

    Do not skip a paragraph simply because it consists of
    "a heading + a definition."

    A single paragraph may contain both:

    - a definition;
    - a normative requirement.

    In such cases, extract ONLY the requirement.
    Do NOT include the definitional portion.

    Inspect EVERY paragraph for multiple independent normative statements.

    If a paragraph contains several separate requirements,
    extract them as separate rules rather than combining them into one.

    === RULES FROM TABLES ===

    Tables in these documents are not merely reference material.

    They are often the PRIMARY source of technical requirements,
    containing specific values for wall thicknesses, pressures, dimensions,
    material grades, and similar parameters.

    Do not limit yourself to textual references such as

    "shall be as given in Table 1".

    If the table itself is present in the provided blocks,
    analyze its rows and extract concrete rules from them.

    How to formulate a rule from a table row:

    - Combine into one sentence:
      - the table title;
      - the key identifying values from that row
        (for example, pressure class, nominal size, trim number);
      - the value from the target column.

    Example:

    If the table "Minimum Wall Thickness" contains the row

    "Class Designation/Nominal Size DN": "25"

    and

    "Minimum Wall Thickness tm mm (in.)/300":
    "6.4 (0.25)"

    then formulate the rule as:

    "For nominal size DN 25 and pressure class 300,
    the minimum wall thickness tm shall be
    6.4 mm (0.25 in.)."

    For table-derived rules, the attribute should generally correspond
    to the table title
    (for example, "Minimum Wall Thickness")
    rather than to an individual column.

    This naturally groups all rows of the same table together.

    Each table row containing meaningful values must become
    a separate rule within the group.

    Do NOT combine multiple rows into one rule.

    Do NOT summarize the table with statements like

    "wall thickness depends on size and pressure class."

    Instead, extract every specific value from every applicable row.

    Do not include empty ("") or "NA" cells in a rule.

    If a table cell contains a symbolic designation or variable
    (for example, a single letter)
    rather than the final value,
    and you find its explanation in nearby blocks
    (paragraphs or notes),

    use the decoded meaning when formulating the rule,
    while optionally preserving the original symbol in parentheses
    for precision.

    === HANDLING PAGE BOUNDARIES ===

    The input contains either ONE or TWO pages.

    If there are TWO pages:

    - the first page is the primary page from which rules must be extracted
      (using all of its block types: paragraphs and tables);

    - the second page is provided ONLY as context
      to determine whether the last statement on the primary page
      is semantically complete or continues onto the next page.

    Do NOT extract rules from the second page itself.

    It will be processed independently in the next step
    when it becomes the primary page.

    If there is only ONE page:

    it is the final page of the document.

    Extract rules from it exactly as you would from any other primary page,
    without any special restrictions caused by the absence of a following page.

    If the last paragraph on the primary page contains a rule
    whose meaning becomes clear only when considering the beginning
    of the next page,

    you must still extract that rule.

    Do not discard it merely because it appears at the end of the page.

    === FINAL REMINDER BEFORE RESPONDING ===

    Before producing your output, verify the following:

    1. Have you analyzed EVERY table row by row and extracted the actual values,
       rather than only the textual references to the table?

    2. Have you inspected definitions in the "Terms and Definitions" section
       (heading + paragraph) for hidden normative requirements,
       rather than extracting only explicitly numbered clauses?

    3. Have you avoided duplicating the same rule under different wording?

    4. If a list of previously identified attributes has been provided,
       have you reused exactly the same attribute names instead of inventing new ones?
    """
    TABLE_EXTRACTION_PROMPT = """Ты извлекаешь технические правила из данных нормативных документов в области нефтегазовой промышленности.
    Тебе может быть дана как таблица так и текст. Если таблица, то в формате JSON: title, columns, rows,
    если текст, то в формате JSON: paragraph, heading.

    Твоя задача — пройтись по КАЖДОЙ строке и КАЖДОМУ значению (кроме пустых/прочерков) и
    сформулировать для каждого отдельное правило. Используй идентифицирующие колонки, 
    чтобы правило было самодостаточным.
    

    ОБЯЗАТЕЛЬНО обработай ВСЕ строки таблицы от первой до последней, без исключений и без
    сокращения количества — не давай "несколько примеров", а обработай таблицу ПОЛНОСТЬЮ.
    Пропускай только значения "—", "NA" или пустые.
    
    Правила ты должен передавать ТОЧЬ В ТОЧЬ, а НЕ ПЕРЕФРАЗИРОВАТЬ своими словами, НЕ ДОДУМЫВАЙ, также не сокращай, 
    старайся задействовать как можно больше полезной информации из данных. Отвечай только на языке оригинала.
    
    После составления правил ты должен их сгруппировать, каждое правило относится к конкретному атрибуту,
    который и будет считаться группирующим. Сам атрибут ты должен сформировать по смыслу из данных что у тебя есть.
    Атрибут - единственное место где ты можешь применять своё воображение и инициативу.

    Верни ТОЛЬКО валидный JSON — объект, где ключ это атрибут, а значение — список правил
для этого атрибута, без markdown, без пояснений:

{"groups": [{"attribute": "...", "rules": ["...", "..."]}]}
    """