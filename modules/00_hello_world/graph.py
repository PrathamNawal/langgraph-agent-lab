"""Module 0: the smallest possible LangGraph agent.

One node, no tools, no branching: just enough wiring to see how a StateGraph
differs from a plain LangChain chain (shared state, explicit edges, a compile step).
"""

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict

load_dotenv()


class State(TypedDict):
    messages: Annotated[list, add_messages]


def build_graph(model: str = "llama-3.1-8b-instant", temperature: float = 0.7, api_key: str | None = None):
    llm = ChatGroq(model=model, temperature=temperature, api_key=api_key or os.environ.get("GROQ_API_KEY"))

    def chat_node(state: State) -> State:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(State)
    graph.add_node("chat", chat_node)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)

    return graph.compile()


if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit("Set GROQ_API_KEY in your .env before running this module.")

    app = build_graph()
    state = {"messages": [{"role": "user", "content": "In one sentence, what is LangGraph?"}]}
    result = app.invoke(state)
    print(result["messages"][-1].content)
