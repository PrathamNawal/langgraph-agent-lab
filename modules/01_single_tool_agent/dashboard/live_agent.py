"""Runs the REAL Module 1 LangGraph agent (modules/01_single_tool_agent/graph.py)
against an eval question, yielding trace events as LangGraph's own .stream()
produces them - so "Try it - Agent" shows the actual graph executing live,
not a reimplementation of its logic.
"""

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from graph import build_graph  # noqa: E402

_NUMBER_RE = re.compile(r"-?\$?\d[\d,]*(\.\d+)?")


def extract_number(text):
    """Pull the first number-looking token out of the model's final answer,
    so it can be compared against the eval question's known ground truth."""
    if not text:
        return None
    m = _NUMBER_RE.search(text)
    if not m:
        return None
    cleaned = m.group(0).replace("$", "").replace(",", "")
    try:
        value = float(cleaned)
        return int(value) if value.is_integer() else value
    except ValueError:
        return None


def run_investigation_stream(item, config, api_key):
    """
    item: {"id", "question", "answer"} from eval_sets/calculator_eval.py
    config: active deployed config dict (model, temperature, system_prompt, enabled_tools)

    Yields trace-event dicts as the real graph executes:
      {"type": "turn", "turn": n}
      {"type": "tool_call", "turn": n, "tool": str, "input": {...}, "result": str}
      {"type": "final", "answer_text": str, "predicted_value": number|None, "turns_used": n}
      {"type": "error", "message": str}
    """
    if not api_key:
        yield {"type": "error", "message": "No Groq API key set. Add it in the sidebar."}
        return
    if not config.get("enabled_tools"):
        yield {"type": "error", "message": "No tools are enabled on the active config."}
        return

    try:
        app = build_graph(
            model=config["model"],
            temperature=config["temperature"],
            api_key=api_key,
            system_prompt=config.get("system_prompt"),
            enabled_tools=config.get("enabled_tools"),
        )
    except Exception as e:
        yield {"type": "error", "message": f"{type(e).__name__}: {e}"}
        return

    state = {"messages": [{"role": "user", "content": item["question"]}]}
    turn = 0
    pending_tool_calls = {}
    final_text = None

    try:
        for update in app.stream(state, stream_mode="updates"):
            if "agent" in update:
                turn += 1
                yield {"type": "turn", "turn": turn}
                ai_msg = update["agent"]["messages"][-1]
                tool_calls = getattr(ai_msg, "tool_calls", None) or []
                if tool_calls:
                    for tc in tool_calls:
                        pending_tool_calls[tc["id"]] = (tc["name"], tc.get("args", {}))
                else:
                    final_text = ai_msg.content
            if "tools" in update:
                for tm in update["tools"]["messages"]:
                    name, args = pending_tool_calls.pop(getattr(tm, "tool_call_id", None), ("tool", {}))
                    yield {"type": "tool_call", "turn": turn, "tool": name, "input": args, "result": tm.content}
    except Exception as e:
        yield {"type": "error", "message": f"{type(e).__name__}: {e}"}
        return

    predicted_value = extract_number(final_text)
    yield {"type": "final", "answer_text": final_text, "predicted_value": predicted_value, "turns_used": turn}
