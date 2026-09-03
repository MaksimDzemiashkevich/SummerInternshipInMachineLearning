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

# Frontend tests (react-scripts / jest)
cd client && npm test                 # watch mode
cd client && npm test -- --testPathPattern=MessageInput   # single test file
```

There are no Python tests. The `structured_documents_graph` module executes the full pipeline at import time (its bottom line calls `graph.invoke(...)`), so running it has side effects.

## Architecture & data flow

There are two separate LangGraph graphs with a shared data artifact between them.

### 1. `structured_documents` — builds the rule knowledge base (offline)

`graphs/structured_documents/structured_documents_graph.py` defines a linear `StateGraph`:

`first_node` → `formatting_text_node` → `grouping_rules_node`

- `first_node` renders each PDF page to an image (PyMuPDF `fitz` + Tesseract OCR for orientation) and sends it to a local Qwen3-VL vision model (`vlm`) to extract structured blocks (paragraphs/headings/tables).
- `formatting_text_node` uses `gpt-4o` to validate/repair table cells and merge paragraphs broken across page boundaries.
- `grouping_rules_node` groups rules by attribute and **appends** them to `graphs/structured_documents/data/output/rule_groups.json` (it loads existing groups first, then re-saves — accumulation, not replace).

Config for this graph lives in `graphs/structured_documents/modules/config_structured_documents_graph.py`, which holds all prompts and paths. It hardcodes a Tesseract binary path (`C:\Program Files\Tesseract-OCR\tesseract.exe`), a local `SentenceTransformer` model (`all-MiniLM-L6-v2`), and an ngrok-hosted VLM endpoint — these are environment-specific and may need changing.

### 2. `rag_agent` — the runtime normalization API

`graphs/rag_agent/modules/config.py` is the entry point for app config:

- `load_rules()` reads `data/input/rules.json`; if absent, it builds it from `structured_documents/data/output/rule_groups.json`.
- `load_vector_store()` loads `data/input/vector_store.json`; if absent, it embeds each rule-group header (`text-embedding-3-large`) into an `InMemoryVectorStore` keyed by attribute and dumps it. **Delete `vector_store.json` (and `rules.json`) to force regeneration** after changing the rules.
- `build_app_config()` assembles `AppConfig` (settings + store + `gpt-4o` chat model) and is cached via `lru_cache` in `api.py`.

`graphs/rag_agent/api.py` exposes `POST /process`, which accepts an uploaded `.xlsx` with `Attribute`/`Value` columns and returns a result `.xlsx`. The LangGraph `StateGraph` (`start_row` → `process_chunk` → `next_row` / `end_node`) iterates over rows; for each attribute it does a similarity search (`RETRIEVAL_K = 4`) to fetch relevant rules, splits them into sentence chunks (`SENTENCES_PER_CHUNK = 10`), and asks `gpt-4o` to normalize the value using `SYSTEM_PROMPT1` (`modules/system_prompts.py`).

- `modules/tools.py` contains unit-conversion `@tool` functions that are defined but **not currently wired into the graph**.
- `modules/system_prompts.py` also has `SYSTEM_PROMPT2` (a routing prompt) that is currently unused.

### Frontend

`client/src` is a minimal create-react-app: `App.js` renders `Layout/ChatPage/ChatPage.jsx`, which composes `Header`, `ChatHistory`, and `MessageInput`. `MessageInput.jsx` posts the selected file as `multipart/form-data` to `http://localhost:8000/process` (hardcoded `API_URL`) via axios, then renders the returned blob as a downloadable `result.xlsx` attachment.

## Conventions / gotchas

- CORS on the backend only allows `http://localhost:3000`.
- `structured_documents_graph.py` opens the PDF via a relative path (`../../graphs/structured_documents/data/input/API-600.pdf`) and imports `Settings` via `from graphs.structured_documents.modules...`, so it must be run from the repo root with `-m`.
- The `.env` file (ignored by git) supplies `OPENAI_API_KEY`.
- Several data outputs (e.g. `graphs/rag_agent/data/output/result*.xlsx`) are generated artifacts and should not be committed.
