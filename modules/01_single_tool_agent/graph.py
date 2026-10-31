"""Module 1: the canonical ReAct-style agent - two tools, a loop, a conditional edge.

Graph shape:

    START -> agent --(has tool call?)--> tools -> agent -> ... -> END
                 \\--(no tool call)----------------------------> END

The LLM decides which tool (calculator or unit_converter) to call, if any. If it does, `tools_condition`
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


_LENGTH_M = {"mm": 0.001, "cm": 0.01, "m": 1.0, "km": 1000.0, "in": 0.0254, "ft": 0.3048, "yd": 0.9144, "mi": 1609.344}
_MASS_KG = {"g": 0.001, "kg": 1.0, "lb": 0.45359237, "oz": 0.028349523125}


@tool
def unit_converter(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a value between units of length (mm, cm, m, km, in, ft, yd, mi), mass
    (g, kg, lb, oz) or temperature (C, F, K). E.g. value=5, from_unit="mi", to_unit="km".
    Use this for any unit conversion instead of doing it yourself."""
    f, t = from_unit.strip().lower(), to_unit.strip().lower()
    try:
        if f in ("c", "f", "k") and t in ("c", "f", "k"):
            celsius = {"c": value, "f": (value - 32) * 5 / 9, "k": value - 273.15}[f]
            result = {"c": celsius, "f": celsius * 9 / 5 + 32, "k": celsius + 273.15}[t]
        else:
            for table in (_LENGTH_M, _MASS_KG):
                if f in table and t in table:
                    result = value * table[f] / table[t]
                    break
            else:
                return f"Error: cannot convert '{from_unit}' to '{to_unit}' (unknown or incompatible units)."
        return f"{value} {from_unit} = {result:.6g} {to_unit}"
    except Exception as e:
        return f"Error converting {value} {from_unit} to {to_unit}: {e}"


TOOLS = [calculator, unit_converter]

SYSTEM_PROMPT = (
    "You are a careful assistant. For ANY arithmetic, no matter how simple it looks "
    "(even single-digit math), you must call the calculator tool rather than computing "
    "it yourself - you are not reliable at mental math and must never guess. For ANY "
    "unit conversion (length, mass, temperature), you must call the unit_converter tool."
)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def build_graph(model: str = "llama-3.1-8b-instant", temperature: float = 0.0, api_key: str | None = None):
    llm = ChatGroq(model=model, temperature=temperature, api_key=api_key or os.environ.get("GROQ_API_KEY"))
    llm_with_tools = llm.bind_tools(TOOLS)

    def agent_node(state: State) -> State:
        messages = state["messages"]
        if not messages or getattr(messages[0], "type", messages[0].get("role") if isinstance(messages[0], dict) else None) != "system":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(messages)
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    graph = StateGraph(State)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))

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
