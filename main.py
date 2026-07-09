from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import re
from langgraph.graph import START, StateGraph, MessagesState, END
from IPython.display import display, Image
import pandas
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Literal


load_dotenv()

embeddings_own = OpenAIEmbeddings(model="text-embedding-3-large")
store = InMemoryVectorStore(embeddings_own)

with open("rules.txt", "r", encoding="utf-8") as file:
    text = file.read()

chunks = text.split("\n\n\n")
docs = []
for chunk in chunks:
    protected = re.sub(r'(?<=\d)\.(?=\d)', '<DOT>', chunk)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZА-Я])', protected)
    sentences = [s.replace('<DOT>', '.').strip() for s in sentences]

    for i in sentences:
        docs.append(Document(i, metadata={"source": "rules.txt"}))

some_document = docs
store.store.clear()
store.add_documents(documents=some_document)

class MessagesState(MessagesState):
    new_data_frame: pandas.DataFrame
    i: int
    attribute: str
    value: str

def start_node(state: MessagesState) -> MessagesState:
    if (data_frame.shape[0] > state["i"]):
        state["attribute"] = data_frame.iloc[state["i"]]["Attribute"]
        state["value"] = data_frame.iloc[state["i"]]["Value"]
    return state;

def llm_node(state: MessagesState) -> Literal["retrieve_context", "end_node"]:
    if state["i"] < data_frame.shape[0]:
        return "retrieve_context"
    else:
        return "end_node"
    # return {"messages": [llm_with_tools.invoke(state.get("messages"))]}

SYSTEM_PROMPT = """
You are a data normalization engine.

Your task is to normalize the provided attribute value using ONLY the supplied technical rules.

General rules:
- Use only the provided rules as the source of normalization decisions.
- Evaluate every retrieved rule before deciding whether it applies.
- Apply every applicable rule to the value.
- Apply substitutions sequentially until all applicable rules have been processed.
- Ignore any line starting with "Reference:" because it is documentation only and must never affect the output.
- If no rule applies, return the original value unchanged.

Substitution rules:
- A rule may express a substitution in different ways, including:
  - "X becomes Y"
  - "X is abbreviated as Y"
  - "Material codes are ..."
  - "The approved short form is ..."
  - "Use code ..."
  Whenever such a rule matches part of the value, replace only that matching part.
- Preserve all other information from the original value unless a rule explicitly changes it.
- Never remove, truncate, reorder, or invent information unless a rule explicitly instructs you to do so.

Formatting rules:
- If a rule demonstrates formatting using an example (for example "GOST-33259"), treat it as a formatting pattern only.
- Never copy literal example values into the output unless they already exist in the original value.
- Never replace a standard number, year, grade, code, or identifier with a different one mentioned elsewhere in the rules.
- Only modify separators, spacing, capitalization, abbreviations, or units when explicitly required by the rules.

Output requirements:
- Return ONLY the final normalized value.
- Do not explain your reasoning.
- Do not summarize the rules.
- Do not output markdown.
- Do not output bullet lists.
- Do not output any extra text.

Examples:

Attribute: Body material
Value: Grey cast iron GG-20
Rules:
Material codes are GCI for grey cast iron, CS for carbon steel, and SS for stainless steel.
Output:
GCI GG-20

Attribute: Seal material
Value: Stainless steel 12Cr18Ni9Ti
Rules:
Material codes are GCI for grey cast iron, CS for carbon steel, and SS for stainless steel.
Output:
SS 12Cr18Ni9Ti

Attribute: End connection
Value: Flanged
Rules:
FLANGED becomes FLG.
Output:
FLG

Attribute: Valve type
Value: Gate valve, wedge
Rules:
GATE VALVE becomes GV.
Output:
GV, wedge
"""

def retrieve_context(state: MessagesState) -> MessagesState:
    """Retrieve information to help answer a query and return list of documents."""
    retrived_documents = store.similarity_search(query = f"{state['attribute']}: {state['value']}", k=8)
    contexts = "\n\n".join(f"{doc.page_content}" for doc in retrived_documents)
    # print(f"Retrieved documents: {retrived_documents}\n\n\nContext documents: {contexts}")
    new_iteration = state.get("i") + 1
    inv = model.invoke([{"role": "system", "content": SYSTEM_PROMPT}, {"role": 'user', "content":
        f"Attribute: {state['attribute']}\nValue: {state['value']}\n\nRelevant rules:\n{contexts}"}])

    new_row = pandas.DataFrame([{"Attribute": state["attribute"],
                                 "Value": state["value"],
                                 "Rules_Value": inv.content}])

    updated_data_frame = pandas.concat([state["new_data_frame"], new_row], ignore_index=True)

    return {"messages": [inv, contexts], "i": new_iteration, "new_data_frame": updated_data_frame}

def end_node(state: MessagesState) -> MessagesState:
    file = state.get("new_data_frame")
    file.to_excel("test_first_phase_table1.xlsx", index=False)
    return state

load_dotenv()


data_frame = pandas.read_excel("attributes.xlsx")

# tools = [retrieve_context]
# tools_node = ToolNode(tools)

model = ChatOpenAI(model="gpt-4o", temperature=0)
# llm_with_tools = model.bind_tools(tools)

builder = StateGraph(MessagesState)
builder.add_node(start_node)
builder.add_node(retrieve_context)
builder.add_node(end_node)
builder.add_edge(START, "start_node")
builder.add_conditional_edges("start_node", llm_node)
builder.add_edge("retrieve_context", "start_node")
builder.add_edge("end_node", END)

graph = builder.compile()

display(Image(graph.get_graph().draw_mermaid_png()))

result = graph.invoke({"messages": [], 'new_data_frame': pandas.DataFrame(
    columns=["Attribute", "Value", "Rules_Value"]), "i":0})

for f in result["messages"]:
    f.pretty_print()