"""Module 2: the same ReAct loop as Module 1, now with four tools and real,
persistent memory via a SQLite checkpointer instead of session-only history.

Graph shape is identical to Module 1 - same agent <-> tools loop, same
tools_condition routing. What's new is entirely in build_graph(): more tools
bound, and a checkpointer passed to .compile() so state survives a process
restart, keyed by thread_id.
"""

import json
import os
import re
import sqlite3
from datetime import date, datetime

import numexpr
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, TypedDict

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHECKPOINT_DB = os.path.join(DATA_DIR, "checkpoints.db")

with open(os.path.join(DATA_DIR, "local_facts.json")) as f:
    _FACTS = json.load(f)


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression, e.g. "2 + 2 * 10" or "(37 * 4) / 3". Use this for
    any arithmetic instead of doing it yourself - you are unreliable at arithmetic."""
    try:
        return str(numexpr.evaluate(expression).item())
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


@tool
def lookup_fact(query: str) -> str:
    """Look up a fact about this project (LangGraph, the modules, the dashboard,
    its tools) from a small local dataset. Use this instead of guessing when asked
    something about how this specific project works."""
    query_words = set(re.findall(r"[a-z0-9]+", query.lower()))
    best_score, best_fact = 0, None
    for fact in _FACTS:
        fact_words = set(fact["keywords"]) | set(re.findall(r"[a-z0-9]+", fact["question"].lower()))
        score = len(query_words & fact_words)
        if score > best_score:
            best_score, best_fact = score, fact
    if best_fact is None:
        return f"No matching fact found for '{query}'."
    return best_fact["answer"]


@tool
def days_between(date1: str, date2: str) -> str:
    """Compute the number of days between two dates, each in YYYY-MM-DD format.
    Use this instead of counting by hand - you are unreliable at date arithmetic."""
    try:
        d1 = date.fromisoformat(date1.strip())
        d2 = date.fromisoformat(date2.strip())
    except ValueError as e:
        return f"Error parsing dates (expected YYYY-MM-DD): {e}"
    return str(abs((d2 - d1).days))


@tool
def web_search(query: str) -> str:
    """Search the live web for current information this agent doesn't otherwise
    have (news, prices, anything time-sensitive). Requires a Tavily API key - if
    none is configured, say so rather than guessing an answer."""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "Web search is unavailable: no TAVILY_API_KEY is configured for this session."
    try:
        from tavily import TavilyClient
    except ImportError:
        return "Web search is unavailable: the 'tavily-python' package isn't installed."
    try:
        client = TavilyClient(api_key=api_key)
        results = client.search(query, max_results=3)
        snippets = [r.get("content", "") for r in results.get("results", [])]
        return "\n---\n".join(snippets) if snippets else "No results found."
    except Exception as e:
        return f"Web search failed: {type(e).__name__}: {e}"


TOOLS = [calculator, lookup_fact, days_between, web_search]

SYSTEM_PROMPT = (
    "You are a careful assistant with four tools: calculator, lookup_fact, "
    "days_between, and web_search. For ANY arithmetic, call calculator. For ANY "
    "question about this project (LangGraph, its modules, its dashboard, its "
    "tools), call lookup_fact. For ANY date-difference question, call days_between. "
    "For anything requiring current/live information, call web_search. Never guess "
    "at something a tool could answer exactly - you are not reliable at any of "
    "these by yourself."
)


class State(TypedDict):
    messages: Annotated[list, add_messages]


_checkpointer_cache = {}


def get_checkpointer(db_path: str = CHECKPOINT_DB) -> SqliteSaver:
    """One shared sqlite3 connection per db path, reused across reruns - this is
    what makes memory persist across a process restart, since it's a real file on
    disk rather than an in-memory structure that dies with the process."""
    if db_path not in _checkpointer_cache:
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        _checkpointer_cache[db_path] = SqliteSaver(conn)
    return _checkpointer_cache[db_path]


def build_graph(
    model: str = "openai/gpt-oss-120b",
    temperature: float = 0.0,
    api_key: str | None = None,
    system_prompt: str | None = None,
    enabled_tools: list[str] | None = None,
    checkpointer: SqliteSaver | None = None,
):
    """enabled_tools restricts which of TOOLS get bound (by name) - a dashboard
    lever that didn't exist in Module 1, since there was only ever one tool to
    enable or disable. checkpointer defaults to the shared on-disk SqliteSaver;
    pass None explicitly (not omitted) only for throwaway/test graphs that
    shouldn't touch the real checkpoint file.
    """
    llm = ChatGroq(model=model, temperature=temperature, api_key=api_key or os.environ.get("GROQ_API_KEY"))
    active_tools = [t for t in TOOLS if enabled_tools is None or t.name in enabled_tools]
    llm_with_tools = llm.bind_tools(active_tools)
    effective_system_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT

    def agent_node(state: State) -> State:
        messages = state["messages"]
        if not messages or getattr(messages[0], "type", messages[0].get("role") if isinstance(messages[0], dict) else None) != "system":
            messages = [{"role": "system", "content": effective_system_prompt}] + list(messages)
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    graph = StateGraph(State)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(active_tools))

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)  # -> "tools" or END
    graph.add_edge("tools", "agent")

    cp = get_checkpointer() if checkpointer is None else checkpointer
    return graph.compile(checkpointer=cp)


if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit("Set GROQ_API_KEY in your .env before running this module.")

    app = build_graph()
    config = {"configurable": {"thread_id": "cli-test"}}
    state = {"messages": [{"role": "user", "content": "What is 847 * 23, and what does this project use for its UI?"}]}
    result = app.invoke(state, config=config)
    for m in result["messages"]:
        print(f"[{m.type}] {m.content}")
