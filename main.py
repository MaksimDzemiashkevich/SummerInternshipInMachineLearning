from langgraph.graph import START, END, StateGraph
from typing import TypedDict
from IPython.display import display, Image
import pandas
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader

class StateOfGraph(TypedDict):
    documents: list[Document]

def indexing_data(state: StateOfGraph):
    text = TextLoader(file_path="rules.txt", encoding="utf-8")
    docs = text.load()
    print(len(docs))
    return {'documents': docs}

data_frame = pandas.read_excel("attributes.xlsx")

builder = StateGraph(StateGraph)
builder.add_node("indexing_data", indexing_data)
builder.add_edge(START, "indexing_data")
builder.add_edge("indexing_data", END)

graph = builder.compile()

# display(Image(graph.get_graph().draw_mermaid_png()))

