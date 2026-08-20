import io
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
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import ToolMessage, RemoveMessage

from .modules.config import AppConfig, Settings, build_app_config
from .modules.system_prompts import SYSTEM_PROMPT1, SYSTEM_PROMPT2
from .modules.tools import array_of_tools

load_dotenv()

# --- логирование ---------------------------------------------------------
logger = logging.getLogger("normalization_pipeline")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(handler)
# ---------------------------------------------------------------------------


class GraphState(MessagesState):
    index_row_table: int
    attribute: str
    value: str
    new_rows: Annotated[list[dict], add]
    data_frame: pd.DataFrame


tools = array_of_tools
tools_node = ToolNode(tools)


def transform_value_loop(state: GraphState) -> GraphState:
    update: dict = {}
    data_frame = state["data_frame"]

    if state["index_row_table"] != 0:
        new_row = {
            "Attribute": state["attribute"],
            "Value": state["value"],
            "Rules_Value": state["messages"][-1].content,
        }
        update["new_rows"] = [new_row]

    if data_frame.shape[0] > state["index_row_table"]:
        update["attribute"] = data_frame.iloc[state["index_row_table"]]["Attribute"]
        update["value"] = data_frame.iloc[state["index_row_table"]]["Value"]
        update["messages"] = [RemoveMessage(id=k.id) for k in state["messages"]]

    return update


def condition_edges_1(state: GraphState) -> Literal["transform_value_row", "end_node"]:
    if state["index_row_table"] < state["data_frame"].shape[0]:
        return "transform_value_row"
    else:
        return "end_node"


def condition_edges_2(state: GraphState) -> Literal["tools", "transform_value_loop"]:
    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, "tool_calls", None)

    # --- лог: вызвала ли модель тул на этом шаге ---
    if tool_calls:
        for call in tool_calls:
            logger.info(
                "  [TOOL CALL] name=%s args=%s",
                call.get("name"),
                call.get("args"),
            )
    else:
        logger.info("  [TOOL CALL] none — модель ответила без вызова инструмента")
    # -------------------------------------------------

    if tool_calls:
        return "tools"
    return "transform_value_loop"


def transform_value_row(state: GraphState, config: RunnableConfig) -> GraphState:
    """Retrieve information to help answer a query and return list of documents."""
    app_config: AppConfig = config["configurable"]["app_config"]
    llm_bind_tools = app_config.model.bind_tools(tools=tools)

    if state["messages"] and isinstance(state["messages"][-1], ToolMessage):
        # --- лог: результат, который вернул тул ---
        logger.info("  [TOOL RESULT] %s", state["messages"][-1].content)
        # -------------------------------------------
        new_inv = llm_bind_tools.invoke(state.get("messages") + [{"role": "system", "content": SYSTEM_PROMPT1}])
        return {"messages": [new_inv]}

    else:
        attribute = state["attribute"]
        value = state["value"]

        logger.info("=" * 80)
        logger.info("Attribute: %r | Value: %r", attribute, value)

        retrieved_documents = app_config.store.similarity_search(query=f"{attribute}", k=2)
        candidate_headers = [doc.page_content for doc in retrieved_documents]
        candidates_text = "\n".join(f"{idx+1}. {header}" for idx, header in enumerate(candidate_headers))

        # --- лог: кандидаты, найденные семантическим поиском (embeddings) ---
        logger.info(
            "  [SEMANTIC SEARCH] top-%d candidates for attribute %r: %s",
            len(candidate_headers), attribute, candidate_headers,
        )
        # -----------------------------------------------------------------

        selection = app_config.router_model.invoke([
            {"role": "system", "content": SYSTEM_PROMPT2},
            {"role": "user", "content": f"Current attribute: {attribute}\n\nCandidate section names:\n{candidates_text}"}
        ])
        selection = selection.content.strip()

        # --- лог: что выбрал router_model из кандидатов ---
        logger.info("  [ROUTER SELECTION] router_model chose section: %r", selection)
        # ----------------------------------------------------

        retrieved_document = next(
            (item for item in retrieved_documents if item.page_content.strip() == selection),
            retrieved_documents[0],
        )

        if retrieved_document.page_content.strip() != selection:
            logger.warning(
                "  [ROUTER SELECTION] router выбрал %r, но точного совпадения среди кандидатов "
                "не нашлось — используется fallback на первый кандидат: %r",
                selection, retrieved_document.page_content,
            )

        contexts = retrieved_document.metadata["text"]

        # --- лог: финальный контекст (правила), переданный в модель нормализации ---
        logger.info("  [CONTEXT USED] section=%r\n%s", retrieved_document.page_content, contexts)
        # -----------------------------------------------------------------------

        new_iteration = state.get("index_row_table") + 1
        inv = llm_bind_tools.invoke([
            {"role": "system", "content": SYSTEM_PROMPT1},
            {"role": "user", "content": f"Attribute: {attribute}\nValue: {value}\n\nRelevant rules:\n{contexts}"},
        ])

        return {
            "messages": [
                {"role": "user", "content": f"Attribute: {attribute}\nValue: {value}\n\nRelevant rules:\n{contexts}"},
                inv,
            ],
            "index_row_table": new_iteration,
        }


def end_node(state: GraphState) -> GraphState:
    return {}


builder = StateGraph(GraphState)
builder.add_node(transform_value_loop)
builder.add_node(transform_value_row)
builder.add_node(end_node)
builder.add_node("tools", tools_node)
builder.add_edge(START, "transform_value_loop")
builder.add_conditional_edges("transform_value_loop", condition_edges_1)
builder.add_conditional_edges("transform_value_row", condition_edges_2)
builder.add_edge("tools", "transform_value_row")
builder.add_edge("end_node", END)

graph = builder.compile()


def run_pipeline(data_frame: pd.DataFrame, app_config: AppConfig) -> pd.DataFrame:
    initial_state = {
        "messages": [],
        "index_row_table": 0,
        "data_frame": data_frame,
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
    app_config: AppConfig = Depends(get_app_config),):

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