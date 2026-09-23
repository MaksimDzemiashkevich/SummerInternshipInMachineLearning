# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A pipeline that normalizes pipeline-valve attribute values (from an Excel sheet) against technical rules extracted from the API-600 standard, exposed through a chat-like web UI. It is built on **LangGraph** state machines on the backend and a **React (create-react-app)** frontend.

## Repo layout

- `graphs/rag_agent/` — the **runtime** RAG pipeline: a FastAPI app (`api.py`) wrapping a LangGraph `StateGraph` that normalizes each row of an uploaded Excel file using rules retrieved from a vector store.
- `graphs/structured_documents/` — an **offline/one-time** extraction pipeline that turns the `API-600.pdf` standard into structured `rule_groups.json` (rules grouped by attribute). It is not served; it runs as a standalone script.
- `client/` — React frontend (chat UI) that uploads an Excel file to the backend and downloads the normalized result.
- `venv2/` — the Python virtualenv with the full backend stack (fastapi, openpyxl, sentence-transformers, pytesseract, langgraph). Use it for all backend commands. (`venv/` is an incomplete second env — don't use it.)

## Commands

All commands run from the repo root unless noted.

```bash
# Backend (FastAPI on http://localhost:8000)
venv2/Scripts/python -m uvicorn graphs.rag_agent.api:app --reload --port 8000

# Frontend (dev server on http://localhost:3000)
cd client && npm install && npm start

# Offline PDF -> rule_groups.json extraction (runs the graph on import)
venv2/Scripts/python -m graphs.structured_documents.structured_documents_graph

# Tag which product/assembly each rule group governs (subjects.json).
# Part of the extraction graph, but also runnable on its own over existing rule_groups.json:
venv2/Scripts/python -m graphs.structured_documents.subject_tagging

# Regression eval of the normalization pipeline (calls the LLM, ~2 min)
venv2/Scripts/python -m graphs.rag_agent.eval.run_eval
venv2/Scripts/python -m graphs.rag_agent.eval.run_eval --rows 6,8,9 --tag alloy

# Frontend tests (react-scripts / jest)
cd client && npm test                 # watch mode
cd client && npm test -- --testPathPattern=MessageInput   # single test file
```

There are no unit tests on the Python side; `graphs/rag_agent/eval` is the regression check for the normalization pipeline (it costs LLM calls, so run it deliberately). The `structured_documents_graph` module executes the full pipeline at import time (its bottom line calls `graph.invoke(...)`), so running it has side effects.

## Architecture & data flow

There are two separate LangGraph graphs with a shared data artifact between them.

### 1. `structured_documents` — builds the rule knowledge base (offline)

`graphs/structured_documents/structured_documents_graph.py` defines a linear `StateGraph`:

`first_node` → `formatting_text_node` → `grouping_rules_node`

- `first_node` renders each PDF page to an image (PyMuPDF `fitz` + Tesseract OCR for orientation) and sends it to a local Qwen3-VL vision model (`vlm`) to extract structured blocks (paragraphs/headings/tables).
- `formatting_text_node` uses `gpt-4o` to validate/repair table cells and merge paragraphs broken across page boundaries.
- `grouping_rules_node` groups rules by attribute and **appends** them to `graphs/structured_documents/data/output/rule_groups.json` (it loads existing groups first, then re-saves — accumulation, not replace).
- `subject_tagging_node` (`subject_tagging.py`) writes `subjects.json`: which product or assembly each group governs. It matters only for a document that covers more than one subject, where the same attribute name means different things; both current documents come out as a single subject (`gate valve`), so the runtime filter below is inert for them. The step is optional — if it fails or the file is missing, everything behaves as before.

Config for this graph lives in `graphs/structured_documents/modules/config_structured_documents_graph.py`, which holds all prompts and paths. Which PDF to extract is taken from `STRUCTURED_DOC_PATH` when that variable is set, so another document can be run without editing the file: `STRUCTURED_DOC_PATH=graphs/structured_documents/data/input/<file>.pdf venv2/Scripts/python -m graphs.structured_documents.structured_documents_graph`. It hardcodes a Tesseract binary path (`C:\Program Files\Tesseract-OCR\tesseract.exe`), a local `SentenceTransformer` model (`all-MiniLM-L6-v2`), and an ngrok-hosted VLM endpoint — these are environment-specific and may need changing.

### 2. `rag_agent` — the runtime normalization API

`graphs/rag_agent/modules/config.py` is the entry point for app config:

- `load_all_rule_sets()` scans `structured_documents/data/output/*/rule_groups.json` and builds one `RuleSet` per document: global rules, local rule groups (name + aliases + body), a semantic index, a BM25 index, and the sidecar files of that document's folder (`deterministic_rules.json`, `phrase_rules.json`, `*_attributes.json`, `excluded_attributes.json`, `render_templates.json`).
- `load_vector_store()` caches each document's index in `data/input/vector_store_<doc>.json`. **Delete that file to force regeneration** after changing the rules of a document.
- `_build_closed_registers()` turns "phrase → code" pairs into closed registers, keyed by code prefix. A pair whose phrase names its own attribute (`Pump size` in the group `pump size`) is a keyword→root pair and is kept out of the registers — `deterministic_order.json` says so when the extraction produced it, `_names_its_attribute()` when it did not, and without that `Pump size 4` normalized to the bare root `PS`; `_attribute_to_register_from_origins()` maps an attribute to its register(s) from where each phrase was extracted. A pair counts as a register entry (not a keyword→root pair) when the phrase is multi-word or the code carries a digit/separator (`_is_register_pair`).
- `_extract_table_pairs()` additionally recovers tables that the offline extraction left as plain rule lines, in all three shapes it leaves them in: `<label>: 'description'; <label>: 'code'`, pipe-separated columns (`Electric motor, squirrel cage | DRV-EM`) and arrows (`Foundation fieldbus -> SIG-FF`). Miss one shape and that document has no closed register at all: nothing is deterministic and nothing is marked pending registration.
- `_render_templates_from_rules()` rebuilds a render template from the document's own units table (`Temperature — Unit: degC; Decimals: 1`) when the extraction produced none, and only when the line's label names the group — a table written per equipment class says nothing about one attribute. Without it, rounding a value already in the right unit was left to the model, which read "already in the prescribed form" and changed nothing. `_match_render_template` then picks, among the retrieved groups' templates, the one whose unit matches the value's own: the first retrieved group is often another quantity. Without it such an attribute has no closed register at all and the model picks the code from prose — which is how a code that exists in no table (`FLG-RTJ`) got invented.
- `hybrid_search()` merges semantic and BM25 hits via RRF and drops groups below `min_score`. `build_rules_text()` then force-includes the group whose name or alias *is* the row's attribute: ranking is a similarity guess, an equal name is not, and without it `IP rating: IP 54` reached the model with four other groups and no IP rule.
- `_build_clause_index()` maps a clause number to its text across the whole document, so a rule that says "see 13.1" can have 13.1 pulled into the prompt even when that clause sits in a group the retrieval did not select.
- `build_app_config()` assembles `AppConfig` (settings + rule sets + chat model) and is cached via `lru_cache` in `api.py`.

`graphs/rag_agent/api.py` exposes three endpoints. `POST /chat` is what the UI talks to: multipart `message` / `history` / `standards` (the names the user picked, empty = all) / `session_id` / optional `file`. It runs the agent (see below) and answers with `{reply, session_id, files}`, where each produced file is base64 inside the JSON, so one round trip carries both the answer and the files. `GET /standards` lists the loaded documents (`name` + a readable `label`) for that picker. `POST /process` accepts an uploaded `.xlsx` with `Attribute`/`Value` columns and returns a result `.xlsx` with one column per rule set (cells needing human review are filled yellow). Any further columns of the upload are passed through to the output and used as row context (`_row_context`): they are never normalized, but they join the retrieval query and the prompts, which is what lets one document's rules for two different subjects ("Item: gate valve" vs "Item: pump") be told apart. Rows with different context are also kept apart by the consistency and reconciliation passes. The rule sets are computed in parallel; rows inside one rule set are processed sequentially by the LangGraph `StateGraph` (`start_row` → `normalize_row` → `next_row` / `end_node`).

The message is handled by an agent (`modules/agent.py`, `AGENT_PROMPT`): a tool-calling loop (`run_agent`, capped by `MAX_AGENT_STEPS`) over the files of the conversation. The tools are `list_files`, `list_standards`, `inspect_file`, `normalize_file` and `edit_result`; the agent never normalizes anything itself, it decides which file, which standards and which of the user's rules go into a run and calls the pipeline. That is what makes "I forgot a rule, run it again" work without re-uploading, and it generalizes: "the first file", "only that standard", "what is in my file?", "fix row 3".

`modules/workspace.py` holds those files — per `session_id`, in process memory, bounded by count and TTL, nothing on disk. A backend restart (including `--reload` on every code edit) forgets them, which is the price of that choice.

The rules a user states never enter the standards' pipeline — the columns of the documents stay exactly what the documents say. They are a separate pass after it (`run_user_rules_column`), applied to the **recorded** value with the user's rules as the only rule text, landing in one extra column (`USER_RULES_COLUMN`). The agent passes them to `normalize_file` per run. A row whose attribute no rule names (`_adhoc_rules_for_attribute`, same word-wise match as everywhere else) is copied through unchanged, so the column shows the effect of the user's rules and nothing else. The rules live for the run they are passed to and are never written to the knowledge base.

Before any of that, `prepare_frame()` makes the upload workable: duplicate headers are renamed (`Attribute (2)`), and `Attribute`/`Value` become text — a datasheet keeps sizes as numbers, test dates as dates and blanks as NaN, all of which used to reach `value.strip()` and fail the request. `normalize_table` then never overwrites an existing column of the upload: a result column whose name is taken gets `(2)` appended (`_free_column_name`). On the way out, `_write_excel_with_highlight` strips characters Excel refuses, caps a cell at 32767 characters, and writes a value starting with `=` as text rather than a formula.

Every value column is followed by a column of the same name plus `APPLIED_SUFFIX`, saying what actually fired on that row: the deterministic step that closed it, the clause the model says it applied (the `precondition` field of `SYSTEM_PROMPT1`, which used to be discarded), the post-passes that changed the string, and the guard that rejected an answer. A row nothing applied to says so. It is assembled from facts, not from the model's prose: what the tools did (`_tool_note`), what changed between the record and the result (`_describe_change` — «добавлен пробел», «пересчёт: 10 kV → 10000 V», «разделители: «-» → «~»»), and only the clause numbers the model cited (`_cited_clauses`). Its own explanation was long, in English and sometimes wrong about itself. That is also how a gap in the document itself is reported — including a citation the extraction cannot resolve (`_unresolved_references`: «normalized per 3.1» when the extractor dropped that clause's number, so its text never reaches the model) — «реестр: такого пункта в документе нет», «правила документа дают два разных ответа», «найдены правила разных изделий, контекст строки не уточняет» — instead of the pipeline inventing a decision.

Per row, `start_row` first tries the deterministic path (`apply_deterministic`): excluded attribute → unchanged; exact phrase match; keyword→root; closed-register match. The root step runs before the registers: `Connector size 18` is a designation, not an entry of the connector-type table that a fuzzy attribute match drags it towards. Only what is left goes to the model.

- The register step (`match_closed_register`) takes an exact/near-exact fuzzy match as the code. In the gray zone it asks the model to pick from the full register (`REGISTER_PROMPT`), which answers with a code, `AMBIGUOUS` or `UNMATCHED` — a reworded entry ("Gate, solid wedge") is matched, a genuinely absent one is flagged rather than approximated.
- A code is only ever written into the cell when the wording matches the register entry by word content (`_wording_matches`): word order, commas, hyphens and a bracketed qualifier may differ, an extra or missing word may not. When the entry is merely the closest one ("NPT threaded end" vs "Threaded end"), the recorded value is reproduced as-is and the row is marked for review — approximating a closed register is exactly what the standards forbid. A value listing several items with `;` is matched item by item, so an enumeration is not a wording difference.
- An outcome of the register is worded `pending registration: <value>` / `ambiguous match: <value>` and carries no code: it used to be prefixed with the code of the *nearest* entry, so a cell read `THRD pending registration: Wafer-type, lugged` — the code of a different connection. In an enumeration, one unresolved item makes the whole cell one verdict plus the recorded value, instead of gluing codes and status words together.
- `apply_deterministic` therefore returns `(value, needs_review)`; the flag travels through the graph as `needs_review_flag` and ends up in the `Ambiguous` column that drives the yellow fill.
- When the retrieved groups belong to different subjects (`subjects.json`), `_select_by_subject` keeps only those matching the row context, comparing word by word so that "gate valves" in the tagging matches "gate valve" in the sheet. With no context, or none matching, every group is kept and the row is marked for review — the alternative is deciding between two subjects' rules by coin flip.
- `_document_covers_row()` leaves the column **empty** (and does not mark it — a blank cell says it itself) when the document does not govern the row's item: a file of pumps run against a valve standard gets blank cells, not copies of the record. The decision is not made from `subjects.json` alone: a kind named in the document's scope but carried by no tagged group (a standard that covers control valves without a single group tagged for them) is answered by asking the model against the opening rules (`SCOPE_PROMPT`), cached per document and context.
- `_equipment_kind()` states in the prompt which kind of equipment the row is about when the row context names it by code rather than by word ("Item: GLV-15001"). The pipeline already worked this out for the subject filter; without passing it on, the model picked a row of the document's own per-equipment table at random (a size in cm came out as "0.1 m", the rule for pipe length).
- `expand_references()` then resolves what those rules point at: clause numbers they cite (two levels deep, capped by `MAX_PULLED_CLAUSES`, and only numbers that the clause index actually defines, so decimals inside values are not mistaken for references), and, when the text mentions a table/appendix/register, the document's own entries for this attribute (`_register_lines_for_attribute`). That function returns **two** tables and they are quoted under different headers: phrase→code is a closed list of values ("use an entry from it"), while keyword→root are roots for the NAME of a designation. Merged into one "closed register" block, the roots became answers: a recorded size of `7,62 cm` came back as `NPS 7,62 cm`. The tables live in the extraction artifacts, not in the rule text — without this step a rule like "use the invented root given in Table 1" reaches the model with no Table 1, and 48 of the 62 roots of XYZ STD 205 appear nowhere in the prose it sees.
- `normalize_row` sends the **whole** relevant rule text at once (global rules + `RETRIEVAL_K = 4` retrieved groups; the query is attribute **and** value). Rules are not fed in chunks: incremental feeding anchored the answer on whatever arrived first.
- Each sample is normalize (`SYSTEM_PROMPT1`) + review (`VERIFY_PROMPT`, catches over- and under-application, example leakage, dropped content). `SELF_CONSISTENCY` samples (default 3, override with `NORMALIZER_SAMPLES`) run in parallel and the majority wins; a split vote marks the row for review.
- `modules/tools.py` also offers `remove_characters` / `replace_substring`, and the prompt tells the model to carry separator edits out with them rather than retyping the value, so the edit lands on every occurrence. `_partial_separator_edit` then checks the result deterministically: if the same character was edited in some places and left in others ("C-Mn-Si" -> "C-MnSi"), the row is marked for review. The check only runs while the value's own content is unchanged — once it has been replaced by a code, its separators are nothing to compare against.
- Post-passes over the finished column: `_strip_redundant_parenthetical` (drops a parenthetical that only repeats what the code already encodes), `_reconcile_equivalent_rows` (`RECONCILE_PROMPT`, unifies how the rules were *read* for values that mean the same thing — it may only change separators or revert to the recorded value, never reorder content), `_apply_render_template` (re-renders a number to the document's decimals/suffix; `_names_clash` keeps a template of a *different* named attribute away — "nominal length" (m, 1 decimal) must not re-render "Nominal size"), `_canonicalize_formatting` (merges results that differ only in whitespace, and only with a strict majority: on a tie it leaves the rows alone rather than inventing a convention — it used to pick the shortest and so dropped a space the document required), `_align_root_spacing` (a root and its number are joined the way the deterministic path joins them — the model wrote `TRIM8` in one row and `TRIM 12` in the next — unless the document's own example keeps the space: `"150 Class" → "Class 150"` means `Class150` is wrong, whatever other documents do with their roots), `_canonicalize_units` (one spelling of a unit per attribute: `3.000 in` and `11.811 in.` in one column — but only with a strict majority, because the case of a unit is meaningful: `Pa` is a pascal and `PA` is a peta-ampere).
- `_rejection_reason()` then rejects an answer the rules cannot justify — one question with four checks behind it, each earned by a concrete failure; the row keeps the recorded value and is marked: `_strip_added_words` (a word that is in neither the record nor the document's tables — `A216 WCB` came back as `ASTM A216 WCB`), `_root_invented` (a root whose designation the record does not name — `F51` → `TRIM51`, `PN 100` → `CL100`; it stands down when the attribute label names the designation and the record's letters are part of its number, so a document may define `10K` → `WP10000`), `_root_misapplied` (a root glued to text instead of a number — `TRIM f6a`), and `_unit_relabelled` (the number unchanged while the unit silently became another one — `9.5` under a `, mm` label came back as `9.50 in.`) and `_unit_invented` (a unit in neither the record nor the label — a bare `150` came back as `150.000 in.`, a claim about the quantity rather than its spelling). `_root_misapplied` also covers a root glued to a measurement (`150 MM` → `DN5.906 in`) or to a decimal (`6 IN` → `NPS6.000`): a designation number is a whole number, not a converted length. Units are compared through pint (`tools.unit_mismatch`), never as strings: `in.`, `in` and `inch` are one unit, and `HB minimum` is not a unit at all.

- The tool loop (`_invoke_with_tools`) is bounded by `MAX_TOOL_STEPS` and refuses to repeat an identical call, and the tools answer `no change: …` when they would do nothing. Without that, a model that kept asking for `replace_substring('m3/h' → 'm3/h')` looped forever: one row ran for five minutes. When the budget runs out the row keeps the recorded value and is marked.
- `modules/tools.py` gives the model one generic `convert_units(quantity, to_unit, decimals)` built on **pint**, not a list of ready-made pairs: with a fixed list the model reaches for the nearest pair when its own is missing (it called inch→mm for a value in cm) or converts in its head. pint parses the value as recorded (decimal comma, `°F`, `6"`, `mm2`), checks dimensionality, and answers `cannot convert: …` for a value with no unit or an impossible conversion — so a failed conversion is visible instead of invented. Tools are wired into the model via `bind_tools` in `modules/config.py`.
- `modules/system_prompts.py` holds the runtime prompts (`SYSTEM_PROMPT1`, `VERIFY_PROMPT`, `REGISTER_PROMPT`, `RECONCILE_PROMPT`, `SUBJECT_CHOICE_PROMPT`, and the agent's `AGENT_PROMPT`). They must stay document-agnostic: no attribute names, values or phrasings of a particular standard.

### 3. `rag_agent/eval` — regression harness

`graphs/rag_agent/eval/expected.py` holds the expected output of both rule sets for every row of `data/input/test_attributes_api600_new_new.xlsx`, the rows that must (and must not) be marked for review, and the row pairs that must produce byte-identical results. `run_eval.py` runs the pipeline and prints accuracy, mismatches and inconsistent pairs. Rows where the source standard is genuinely ambiguous are documented there.

### Frontend

`client/src` is a minimal create-react-app: `App.js` renders `Layout/ChatPage/ChatPage.jsx`, which composes `Header`, `ChatHistory`, and `MessageInput`. `MessageInput.jsx` posts the message, the last `HISTORY_LIMIT` turns, the picked standards (from `GET /standards`) and the attached file as `multipart/form-data` to `http://localhost:8000/chat` (hardcoded `API_BASE`) via axios, prints the reply and turns the base64 of each returned file into a downloadable attachment. It generates a `session_id` per page load (that is the conversation whose files the backend keeps) and renders every file of the `files` array as an attachment. The send button is rendered by hand (`footer`): the stock one is disabled on an empty input, and a file with no covering text must still be sendable; while a request is in flight it is the library's `LoadingButton`, whose `onCancel` aborts the axios request (the backend, being synchronous, still finishes the run).

## Conventions / gotchas

- CORS on the backend only allows `http://localhost:3000`.
- `structured_documents_graph.py` opens the PDF via a relative path (`../../graphs/structured_documents/data/input/API-600.pdf`) and imports `Settings` via `from graphs.structured_documents.modules...`, so it must be run from the repo root with `-m`.
- The `.env` file (ignored by git) supplies `OPENAI_API_KEY`.
- Several data outputs (e.g. `graphs/rag_agent/data/output/result*.xlsx`) are generated artifacts and should not be committed.
