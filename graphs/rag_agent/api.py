import io
import re
import logging
from functools import lru_cache
from typing import Literal, Annotated
from operator import add

import pandas as pd
from dotenv import load_dotenv

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Depends, UploadFile, File
from fastapi.responses import Response

from langgraph.graph import START, StateGraph, MessagesState, END
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import RemoveMessage, ToolMessage

from .modules.config import AppConfig, Settings, build_app_config
from .modules.system_prompts import SYSTEM_PROMPT1
from .modules.tools import array_of_tools

TOOLS_BY_NAME = {t.name: t for t in array_of_tools}

load_dotenv()

logger = logging.getLogger("normalization_pipeline")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(handler)

SENTENCES_PER_CHUNK = 10
RETRIEVAL_K = 4


class GraphState(MessagesState):
    index_row_table: int
    attribute: str
    value: str
    new_rows: Annotated[list[dict], add]
    data_frame: pd.DataFrame
    remaining_chunks: list[str]


def split_into_sentence_chunks(text: str, sentences_per_chunk: int = SENTENCES_PER_CHUNK) -> list[str]:

    if not text:
        return []

    raw_units = re.split(r'(?<=[.!?])\s+|\n+', text)
    units = [u.strip() for u in raw_units if u.strip()]

    if not units:
        return []

    chunks = []
    for i in range(0, len(units), sentences_per_chunk):
        chunk_text = "\n".join(units[i:i + sentences_per_chunk])
        chunks.append(chunk_text)

    return chunks


def start_row(state: GraphState, config: RunnableConfig) -> GraphState:

    update: dict = {}
    data_frame = state["data_frame"]
    app_config: AppConfig = config["configurable"]["app_config"]

    if state["index_row_table"] != 0 and state.get("messages"):
        last_answer = state["messages"][-1].content
        new_row = {
            "Attribute": state["attribute"],
            "Value": state["value"],
            "Rules_Value": last_answer,
        }
        update["new_rows"] = [new_row]

    if state["index_row_table"] < data_frame.shape[0]:
        attribute = data_frame.iloc[state["index_row_table"]]["Attribute"]
        value = data_frame.iloc[state["index_row_table"]]["Value"]

        update["attribute"] = attribute
        update["value"] = value
        update["messages"] = [RemoveMessage(id=k.id) for k in state["messages"]]

        retrieved_documents = app_config.store.similarity_search(query=f"{attribute}", k=RETRIEVAL_K)

        all_chunks: list[str] = []
        for idx, doc in enumerate(retrieved_documents):
            doc_text = doc.metadata.get("text", doc.page_content)
            doc_chunks = split_into_sentence_chunks(doc_text, SENTENCES_PER_CHUNK)
            all_chunks.extend(doc_chunks)
            logger.info(
                "  [SEARCH] Doc %d/%d (%r): %r",
                idx + 1, len(retrieved_documents), doc.page_content[:], doc_text
            )

        if not all_chunks:
            all_chunks = ["No relevant rules were found in the knowledge base for this attribute."]

        update["remaining_chunks"] = all_chunks
        logger.info("=" * 80)
        logger.info(
            "Row %d/%d | Attribute: %r | Value: %r | Всего чанков со всех 4 документов: %d",
            state["index_row_table"] + 1, data_frame.shape[0], attribute, value, len(all_chunks)
        )

    return update


def process_chunk(state: GraphState, config: RunnableConfig) -> GraphState:
    app_config: AppConfig = config["configurable"]["app_config"]

    chunks = state["remaining_chunks"]
    current_chunk, rest_chunks = chunks[0], chunks[1:]

    is_first_chunk = len(state["messages"]) == 0

    if is_first_chunk:
        user_content = (
            f"Attribute: {state['attribute']}\n"
            f"Value: {state['value']}\n\n"
            f"Relevant rules:\n{current_chunk}"
        )
    else:
        user_content = (
            f"Additional relevant rules for the same attribute/value:\n{current_chunk}\n\n"
            f"If any of these rules apply, update your previously normalized value accordingly. "
            f"Otherwise keep your previous answer unchanged."
        )

    user_msg = {"role": "user", "content": user_content}

    system_prompt = SYSTEM_PROMPT1
    if app_config.global_rules:
        system_prompt = (
            SYSTEM_PROMPT1
            + "\n\nGlobal rules (apply to every attribute):\n"
            + app_config.global_rules
        )

    messages = [{"role": "system", "content": system_prompt}] + state["messages"] + [user_msg]

    response = app_config.model.invoke(messages)
    while getattr(response, "tool_calls", None):
        tool_messages = []
        for tc in response.tool_calls:
            tool_fn = TOOLS_BY_NAME.get(tc["name"])
            if tool_fn is None:
                content = "unknown tool"
            else:
                try:
                    content = str(tool_fn.invoke(tc["args"]))
                except Exception as e:
                    content = f"error: {e}"
            tool_messages.append(ToolMessage(content=content, tool_call_id=tc["id"]))
        messages = messages + [response] + tool_messages
        response = app_config.model.invoke(messages)

    return {
        "messages": [user_msg, response],
        "remaining_chunks": rest_chunks,
    }


def next_row(state: GraphState) -> GraphState:
    return {"index_row_table": state["index_row_table"] + 1}


def condition_after_start_row(state: GraphState) -> Literal["process_chunk", "end_node"]:
    if state["index_row_table"] < state["data_frame"].shape[0]:
        return "process_chunk"
    return "end_node"


def condition_after_process_chunk(state: GraphState) -> Literal["process_chunk", "next_row"]:
    if state["remaining_chunks"]:
        return "process_chunk"
    return "next_row"


def end_node(state: GraphState) -> GraphState:
    return {}


builder = StateGraph(GraphState)

builder.add_node("start_row", start_row)
builder.add_node("process_chunk", process_chunk)
builder.add_node("next_row", next_row)
builder.add_node("end_node", end_node)

builder.add_edge(START, "start_row")
builder.add_conditional_edges("start_row", condition_after_start_row)
builder.add_conditional_edges("process_chunk", condition_after_process_chunk)
builder.add_edge("next_row", "start_row")
builder.add_edge("end_node", END)

graph = builder.compile()


def run_pipeline(data_frame: pd.DataFrame, app_config: AppConfig) -> pd.DataFrame:
    initial_state = {
        "messages": [],
        "index_row_table": 0,
        "data_frame": data_frame,
        "remaining_chunks": [],
        "new_rows": [],
    }
    final_state = graph.invoke(
        initial_state,
        config={"configurable": {"app_config": app_config}},
    )
    return pd.DataFrame(final_state["new_rows"])


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache
def get_app_config() -> AppConfig:
    return build_app_config(Settings())


@app.post("/process")
def process_excel(
        file: UploadFile = File(...),
        app_config: AppConfig = Depends(get_app_config),
):
    contents = file.file.read()
    data_frame = pd.read_excel(io.BytesIO(contents))

    result_df = run_pipeline(data_frame, app_config)

    output_buffer = io.BytesIO()
    result_df.to_excel(output_buffer, index=False)
    output_buffer.seek(0)

    return Response(
        content=output_buffer.read(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=result.xlsx"},
    )