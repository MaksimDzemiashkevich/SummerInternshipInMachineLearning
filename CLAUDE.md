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

Config for this graph lives in `graphs/structured_documents/modules/config_structured_documents_graph.py`, which holds all prompts and paths. It hardcodes a Tesseract binary path (`C:\Program Files\Tesseract-OCR\tesseract.exe`), a local `SentenceTransformer` model (`all-MiniLM-L6-v2`), and an ngrok-hosted VLM endpoint — these are environment-specific and may need changing.

### 2. `rag_agent` — the runtime normalization API

`graphs/rag_agent/modules/config.py` is the entry point for app config:

- `load_all_rule_sets()` scans `structured_documents/data/output/*/rule_groups.json` and builds one `RuleSet` per document: global rules, local rule groups (name + aliases + body), a semantic index, a BM25 index, and the sidecar files of that document's folder (`deterministic_rules.json`, `phrase_rules.json`, `*_attributes.json`, `excluded_attributes.json`, `render_templates.json`).
- `load_vector_store()` caches each document's index in `data/input/vector_store_<doc>.json`. **Delete that file to force regeneration** after changing the rules of a document.
- `_build_closed_registers()` turns "phrase → code" pairs into closed registers, keyed by code prefix; `_attribute_to_register_from_origins()` maps an attribute to its register(s) from where each phrase was extracted. A pair counts as a register entry (not a keyword→root pair) when the phrase is multi-word or the code carries a digit/separator (`_is_register_pair`).
- `_extract_table_pairs()` additionally recovers tables that the offline extraction left as plain rule lines (`<label>: 'description'; <label>: 'code'`, e.g. Table 2 of XYZ STD 100). Without it such an attribute has no closed register at all and the model picks the code from prose — which is how a code that exists in no table (`FLG-RTJ`) got invented.
- `hybrid_search()` merges semantic and BM25 hits via RRF and drops groups below `min_score`.
- `_build_clause_index()` maps a clause number to its text across the whole document, so a rule that says "see 13.1" can have 13.1 pulled into the prompt even when that clause sits in a group the retrieval did not select.
- `build_app_config()` assembles `AppConfig` (settings + rule sets + chat model) and is cached via `lru_cache` in `api.py`.

`graphs/rag_agent/api.py` exposes two endpoints. `POST /chat` is what the UI talks to: multipart `message` / `history` / optional `file`. Without a file it is an ordinary conversation (`modules/chat.py` + `CHAT_PROMPT`); with a file it normalizes it and returns the result `.xlsx` base64-encoded inside the JSON reply, so one round trip carries both the answer and the file. `POST /process` accepts an uploaded `.xlsx` with `Attribute`/`Value` columns and returns a result `.xlsx` with one column per rule set (cells needing human review are filled yellow). Any further columns of the upload are passed through to the output and used as row context (`_row_context`): they are never normalized, but they join the retrieval query and the prompts, which is what lets one document's rules for two different subjects ("Item: gate valve" vs "Item: pump") be told apart. Rows with different context are also kept apart by the consistency and reconciliation passes. The rule sets are computed in parallel; rows inside one rule set are processed sequentially by the LangGraph `StateGraph` (`start_row` → `normalize_row` → `next_row` / `end_node`).

The message sent with the file is not decoration: `extract_adhoc_rules` (`RULE_EXTRACTION_PROMPT`) pulls out the rules the user states in it, and they travel through the graph as `extra_rules`. They apply to that request only — nothing is written to the knowledge base — and are appended to the retrieved rule text as a block that outranks the document where the two conflict (`format_adhoc_rules`). A rule names an attribute (matched against the row by `_attribute_matches`) or, with no attribute, governs every row. A row an ad-hoc rule touches never ends on the deterministic shortcut, which the model never sees: the deterministic answer is computed anyway and handed to the model as the starting value, so the row keeps the document's precision and still gets the user's rule.

Per row, `start_row` first tries the deterministic path (`apply_deterministic`): excluded attribute → unchanged; exact phrase match; closed-register match; keyword→root. Only what is left goes to the model.

- The register step (`match_closed_register`) takes an exact/near-exact fuzzy match as the code. In the gray zone it asks the model to pick from the full register (`REGISTER_PROMPT`), which answers with a code, `AMBIGUOUS` or `UNMATCHED` — a reworded entry ("Gate, solid wedge") is matched, a genuinely absent one is flagged rather than approximated.
- A code is only ever written into the cell when the wording matches the register entry by word content (`_wording_matches`): word order, commas, hyphens and a bracketed qualifier may differ, an extra or missing word may not. When the entry is merely the closest one ("NPT threaded end" vs "Threaded end"), the recorded value is reproduced as-is and the row is marked for review — approximating a closed register is exactly what the standards forbid. A value listing several items with `;` is matched item by item, so an enumeration is not a wording difference.
- `apply_deterministic` therefore returns `(value, needs_review)`; the flag travels through the graph as `needs_review_flag` and ends up in the `Ambiguous` column that drives the yellow fill.
- When the retrieved groups belong to different subjects (`subjects.json`), `_select_by_subject` keeps only those matching the row context, comparing word by word so that "gate valves" in the tagging matches "gate valve" in the sheet. With no context, or none matching, every group is kept and the row is marked for review — the alternative is deciding between two subjects' rules by coin flip.
- `expand_references()` then resolves what those rules point at: clause numbers they cite (two levels deep, capped by `MAX_PULLED_CLAUSES`, and only numbers that the clause index actually defines, so decimals inside values are not mistaken for references), and, when the text mentions a table/appendix/register, the document's own entries for this attribute (`_register_lines_for_attribute`). The tables live in the extraction artifacts, not in the rule text — without this step a rule like "use the invented root given in Table 1" reaches the model with no Table 1, and 48 of the 62 roots of XYZ STD 205 appear nowhere in the prose it sees.
- `normalize_row` sends the **whole** relevant rule text at once (global rules + `RETRIEVAL_K = 4` retrieved groups; the query is attribute **and** value). Rules are not fed in chunks: incremental feeding anchored the answer on whatever arrived first.
- Each sample is normalize (`SYSTEM_PROMPT1`) + review (`VERIFY_PROMPT`, catches over- and under-application, example leakage, dropped content). `SELF_CONSISTENCY` samples (default 3, override with `NORMALIZER_SAMPLES`) run in parallel and the majority wins; a split vote marks the row for review.
- `modules/tools.py` also offers `remove_characters` / `replace_substring`, and the prompt tells the model to carry separator edits out with them rather than retyping the value, so the edit lands on every occurrence. `_partial_separator_edit` then checks the result deterministically: if the same character was edited in some places and left in others ("C-Mn-Si" -> "C-MnSi"), the row is marked for review. The check only runs while the value's own content is unchanged — once it has been replaced by a code, its separators are nothing to compare against.
- Post-passes over the finished column: `_strip_redundant_parenthetical` (drops a parenthetical that only repeats what the code already encodes), `_reconcile_equivalent_rows` (`RECONCILE_PROMPT`, unifies how the rules were *read* for values that mean the same thing — it may only change separators or revert to the recorded value, never reorder content), `_apply_render_template` (re-renders a number to the document's decimals/suffix), `_canonicalize_formatting` (merges results that differ only in whitespace).

- `modules/tools.py` contains unit-conversion `@tool` functions, wired into the model via `bind_tools` in `modules/config.py`.
- `modules/system_prompts.py` holds the runtime prompts (`SYSTEM_PROMPT1`, `VERIFY_PROMPT`, `REGISTER_PROMPT`, `RECONCILE_PROMPT`, `SUBJECT_CHOICE_PROMPT`, and the conversational `CHAT_PROMPT` / `RULE_EXTRACTION_PROMPT`). They must stay document-agnostic: no attribute names, values or phrasings of a particular standard.

### 3. `rag_agent/eval` — regression harness

`graphs/rag_agent/eval/expected.py` holds the expected output of both rule sets for every row of `data/input/test_attributes_api600_new_new.xlsx`, the rows that must (and must not) be marked for review, and the row pairs that must produce byte-identical results. `run_eval.py` runs the pipeline and prints accuracy, mismatches and inconsistent pairs. Rows where the source standard is genuinely ambiguous are documented there.

### Frontend

`client/src` is a minimal create-react-app: `App.js` renders `Layout/ChatPage/ChatPage.jsx`, which composes `Header`, `ChatHistory`, and `MessageInput`. `MessageInput.jsx` posts the message, the last `HISTORY_LIMIT` turns and the attached file (all three optional, but not all three empty) as `multipart/form-data` to `http://localhost:8000/chat` (hardcoded `API_URL`) via axios, prints the reply and, when the JSON carries a file, turns its base64 into a downloadable `result.xlsx` attachment. The send button is rendered by hand (`footer`): the stock one is disabled on an empty input, and a file with no covering text must still be sendable.

## Conventions / gotchas

- CORS on the backend only allows `http://localhost:3000`.
- `structured_documents_graph.py` opens the PDF via a relative path (`../../graphs/structured_documents/data/input/API-600.pdf`) and imports `Settings` via `from graphs.structured_documents.modules...`, so it must be run from the repo root with `-m`.
- The `.env` file (ignored by git) supplies `OPENAI_API_KEY`.
- Several data outputs (e.g. `graphs/rag_agent/data/output/result*.xlsx`) are generated artifacts and should not be committed.
