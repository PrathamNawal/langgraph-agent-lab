"""Module 2: chat UI with four tools and REAL persistent memory.

Unlike Module 1 (which resent the full st.session_state.messages list every
turn, and lost everything on refresh), this app only ever sends the new user
message - the SQLite checkpointer merges it with persisted history by
thread_id. Refresh the page, or restart `streamlit run` entirely, and typing
the same Conversation ID in the sidebar resumes exactly where you left off.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common.groq_ui import groq_sidebar

sys.path.insert(0, str(Path(__file__).parent))
from graph import build_graph, TOOLS

st.set_page_config(page_title="Module 2: multi-tool agent with memory", page_icon=":material/memory:")

st.title("Module 2: multi-tool agent with real memory")
st.caption("Four tools, and a SQLite checkpointer instead of session-only history.")

api_key, model = groq_sidebar()

with st.sidebar:
    st.divider()
    thread_id = st.text_input(
        "Conversation ID",
        value="default-thread",
        help="Restart this app and re-enter the same ID to resume this exact conversation - "
             "that's the SQLite checkpointer, not browser state.",
    )
    tool_names = [t.name for t in TOOLS]
    enabled_tools = st.multiselect("Enabled tools", tool_names, default=tool_names)

if not enabled_tools:
    st.warning("Enable at least one tool in the sidebar.", icon=":material/build:")
    st.stop()

# Building the graph and reading persisted history need no API key - only
# actually calling the LLM does. So history for this Conversation ID renders
# even before a key is entered, which is itself evidence the memory survived
# independently of credentials, not just of the browser session.
app = build_graph(model=model or "openai/gpt-oss-120b", api_key=api_key or "unset", enabled_tools=enabled_tools)
config = {"configurable": {"thread_id": thread_id}}


def build_turns(messages):
    """Group the flat message list into user turns and assistant turns, each
    assistant turn carrying the tool calls that led to its final answer -
    recomputed fresh from the checkpointed state every rerun, not cached in
    session state the way Module 1 had to."""
    turns = []
    pending_trace = []
    for m in messages:
        mtype = getattr(m, "type", None)
        if mtype == "human":
            turns.append({"role": "user", "content": m.content})
        elif mtype == "ai":
            tool_calls = getattr(m, "tool_calls", None) or []
            if tool_calls:
                for tc in tool_calls:
                    pending_trace.append({"name": tc["name"], "args": tc["args"], "result": None})
            elif m.content:
                turns.append({"role": "assistant", "content": m.content, "trace": pending_trace})
                pending_trace = []
        elif mtype == "tool":
            for call in reversed(pending_trace):
                if call["result"] is None:
                    call["result"] = m.content
                    break
    return turns


def render_turn(turn):
    with st.chat_message(turn["role"]):
        st.write(turn["content"])
        if turn["role"] == "assistant" and turn.get("trace"):
            with st.expander("Tool calls made for this answer", icon=":material/build:"):
                for call in turn["trace"]:
                    arg_str = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
                    st.code(f"{call['name']}({arg_str}) -> {call['result']}", language="text")


existing_state = app.get_state(config)
history = existing_state.values.get("messages", []) if existing_state.values else []
for turn in build_turns(history):
    render_turn(turn)

if not api_key:
    st.info("Enter your Groq API key in the sidebar to send a new message.", icon=":material/key:")
    st.stop()

if prompt := st.chat_input("Try: \"What's 847 * 23, and how many days between 2026-01-01 and 2026-03-15?\""):
    with st.chat_message("user"):
        st.write(prompt)

    with st.spinner("Thinking..."):
        try:
            result = app.invoke({"messages": [{"role": "user", "content": prompt}]}, config=config)
        except Exception as e:
            st.error(
                f"This model ({model}) couldn't run the agent - likely no tool-calling "
                f"support. Pick a different model in the sidebar and try again.\n\n`{e}`"
            )
            st.stop()

    new_turns = build_turns(result["messages"])
    render_turn(new_turns[-1])

with st.expander("What's happening under the hood"):
    st.markdown(
        "- The sidebar's **Conversation ID** is the `thread_id` passed to every `.invoke()` call\n"
        "- Only the new user message is ever sent in - the `SqliteSaver` checkpointer merges it "
        "with whatever's already persisted for this thread_id, via the same `add_messages` reducer "
        "from Module 1\n"
        "- `app.get_state(config)` reads the persisted state back out, which is how the chat "
        "history above survives a full restart of this process, not just a page refresh\n"
        "- Unchecking a tool in the sidebar rebuilds the graph with fewer tools bound - the agent "
        "genuinely can't call a disabled tool, it's not just hidden from the trace"
    )
