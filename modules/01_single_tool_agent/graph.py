"""Module 1: the canonical ReAct-style agent - one tool, a loop, a conditional edge.

Graph shape:

    START -> agent --(has tool call?)--> tools -> agent -> ... -> END
                 \\--(no tool call)----------------------------> END

The LLM decides whether to call the calculator tool. If it does, `tools_condition`
routes to the `tools` node, which executes it and loops back to `agent`. If not,
the graph ends. This loop is what makes it an *agent* rather than a single LLM call.
"""

import os

import numexpr
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, TypedDict

load_dotenv()


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression, e.g. "2 + 2 * 10" or "(37 * 4) / 3". Use this for
    any arithmetic instead of doing it yourself - you are unreliable at arithmetic."""
    try:
        return str(numexpr.evaluate(expression).item())
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


# Base unit per category: length in meters, weight in kilograms. To convert,
# go unit -> base -> unit, so adding a new unit only ever needs one number.
_UNIT_GROUPS = {
    "length": {"m": 1.0, "km": 1000.0, "mi": 1609.344, "ft": 0.3048, "cm": 0.01},
    "weight": {"kg": 1.0, "g": 0.001, "lb": 0.453592, "oz": 0.0283495},
}


@tool
def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a numeric value between units of length (m, km, mi, ft, cm) or
    weight (kg, g, lb, oz). Use this for any unit conversion instead of doing
    it yourself - you are unreliable at remembering exact conversion factors."""
    from_unit = from_unit.strip().lower()
    to_unit = to_unit.strip().lower()

    group = next((g for g in _UNIT_GROUPS.values() if from_unit in g and to_unit in g), None)
    if group is None:
        known = {u for g in _UNIT_GROUPS.values() for u in g}
        return f"Error: can't convert '{from_unit}' to '{to_unit}' - both units must be from the same category. Known units: {sorted(known)}"

    base_value = value * group[from_unit]
    result = base_value / group[to_unit]
    return str(result)


TOOLS = [calculator, convert_units]

SYSTEM_PROMPT = (
    "You are a careful assistant. For ANY arithmetic, no matter how simple it looks "
    "(even single-digit math), you must call the calculator tool rather than computing "
    "it yourself - you are not reliable at mental math and must never guess. For ANY "
    "unit conversion, you must call the convert_units tool rather than recalling the "
    "conversion factor from memory - you are not reliable at that either."
)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def build_graph(
    model: str = "llama-3.1-8b-instant",
    temperature: float = 0.0,
    api_key: str | None = None,
    system_prompt: str | None = None,
    enabled_tools: list[str] | None = None,
):
    """system_prompt lets a caller (e.g. the PM dashboard) override the default
    SYSTEM_PROMPT to experiment with its wording - the dashboard's "Play" page
    deploys configs that change this, so Module 1's own app.py (which never
    passes it) keeps behaving exactly as before. enabled_tools restricts which
    of TOOLS get bound (by name); None means all of them."""
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

    # metadata={"kind": ..., "configurable": [...]} is a built-in LangGraph
    # feature (shows up on get_graph().nodes[id].metadata) - the dashboard's
    # visual canvas (common/agent_canvas) reads this directly off the real
    # compiled graph instead of maintaining a separate node registry, so the
    # canvas can never drift out of sync with what this graph actually does.
    graph = StateGraph(State)
    graph.add_node("agent", agent_node, metadata={"kind": "agent", "configurable": ["system_prompt", "model", "temperature"]})
    graph.add_node("tools", ToolNode(active_tools), metadata={"kind": "tool_router", "configurable": ["enabled_tools"]})

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)  # -> "tools" or END
    graph.add_edge("tools", "agent")

    return graph.compile()


if __name__ == "__main__":
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit("Set GROQ_API_KEY in your .env before running this module.")

    app = build_graph()
    state = {"messages": [{"role": "user", "content": "What is 847 * 23, plus 19?"}]}
    result = app.invoke(state)
    for m in result["messages"]:
        print(f"[{m.type}] {m.content}")
