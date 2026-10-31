"""Runs the REAL Module 2 LangGraph agent (modules/02_multi_tool_memory/graph.py)
against an eval question, yielding trace events as LangGraph's own .stream()
produces them - so "Try it - Agent" shows the actual graph executing live,
not a reimplementation of its logic.

Each investigation gets a fresh, unique thread_id (not the real app's shared
Conversation ID) so dashboard eval runs never read or pollute an actual
user's persisted conversation, and so config comparisons stay apples-to-apples
(every run starts from a genuinely blank memory state).
"""

import os
import re
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from graph import build_graph  # noqa: E402

_NUMBER_RE = re.compile(r"-?\$?\d[\d,]*(\.\d+)?")


def extract_number(text):
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


def score_answer(answer_text, expected):
    """Scores a free-text answer against an eval item's expected value. Numeric
    expected values (calculator, days_between items) are compared as numbers
    with tolerance; text expected values (lookup_fact items) are compared as a
    case-insensitive substring match, since the model may add surrounding
    words around the exact fact."""
    if answer_text is None:
        return False
    try:
        expected_num = float(str(expected).replace(",", ""))
        predicted_num = extract_number(answer_text)
        return predicted_num is not None and abs(predicted_num - expected_num) < 0.01
    except ValueError:
        pass
    normalize = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()
    return normalize(str(expected)) in normalize(answer_text)


def run_investigation_stream(item, config, api_key):
    """
    item: {"id", "question", "answer", "tool"} from eval_sets/module_02.jsonl
    config: active deployed config dict (system_prompt, model, temperature,
            enabled_tools, memory_window)

    Yields trace-event dicts as the real graph executes:
      {"type": "turn", "turn": n}
      {"type": "tool_call", "turn": n, "tool": str, "input": {...}, "result": str}
      {"type": "final", "answer_text": str, "turns_used": n}
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
            memory_window=config.get("memory_window"),
        )
    except Exception as e:
        yield {"type": "error", "message": f"{type(e).__name__}: {e}"}
        return

    run_config = {"configurable": {"thread_id": f"dashboard-eval-{uuid.uuid4()}"}}
    turn = 0
    pending_tool_calls = {}
    final_text = None

    try:
        for update in app.stream({"messages": [{"role": "user", "content": item["question"]}]}, config=run_config, stream_mode="updates"):
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

    yield {"type": "final", "answer_text": final_text, "turns_used": turn}
