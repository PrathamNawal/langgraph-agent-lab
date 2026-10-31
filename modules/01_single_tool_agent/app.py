"""Module 1: chat UI that also shows the agent's tool-call trace.

Still a throwaway UI, not the dashboard (Module 2) - but this one surfaces what
Module 1 is actually teaching: watch the agent decide to call a tool, see the
tool's real result, and see it loop back before answering.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from common.groq_ui import groq_sidebar

sys.path.insert(0, str(Path(__file__).parent))
from graph import build_graph

st.set_page_config(page_title="Module 1: single tool agent", page_icon=":material/calculate:")

st.title("Module 1: single tool-using agent")
st.caption("A ReAct-style loop: the agent decides whether to call the calculator tool.")

api_key, model = groq_sidebar()

if not api_key:
    st.info("Enter your Groq API key in the sidebar to start chatting.", icon=":material/key:")
    st.stop()

app = build_graph(model=model, api_key=api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "traces" not in st.session_state:
    st.session_state.traces = []

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant" and i < len(st.session_state.traces) and st.session_state.traces[i]:
            with st.expander("Tool calls made for this answer", icon=":material/build:"):
                for call in st.session_state.traces[i]:
                    arg_str = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
                    st.code(f"{call['name']}({arg_str}) -> {call['result']}", language="text")

if prompt := st.chat_input("Try: \"What is 847 * 23, plus 19?\""):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                result = app.invoke({"messages": st.session_state.messages})
        except Exception as e:
            st.error(
                f"This model ({model}) couldn't run the agent - likely no tool-calling "
                f"support. Pick a different model in the sidebar and try again.\n\n`{e}`"
            )
            st.session_state.messages.pop()  # drop the user turn so retry is clean
            st.stop()

        full_messages = result["messages"]
        trace = []
        for m in full_messages:
            if getattr(m, "type", None) == "ai" and getattr(m, "tool_calls", None):
                for tc in m.tool_calls:
                    matching = next(
                        (fm for fm in full_messages if getattr(fm, "type", None) == "tool" and fm.tool_call_id == tc["id"]),
                        None,
                    )
                    trace.append({
                        "name": tc["name"],
                        "args": tc["args"],
                        "result": matching.content if matching else "?",
                    })

        response = full_messages[-1].content
        st.write(response)
        if trace:
            with st.expander("Tool calls made for this answer", icon=":material/build:"):
                for call in trace:
                    arg_str = ", ".join(f"{k}={v!r}" for k, v in call["args"].items())
                    st.code(f"{call['name']}({arg_str}) -> {call['result']}", language="text")

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.traces.append(trace)

with st.expander("What's happening under the hood"):
    st.markdown(
        "- `agent` node calls the LLM with `calculator` bound via `bind_tools`\n"
        "- `tools_condition` checks the LLM's last message for a tool call and routes to "
        "`tools` (a `ToolNode`) or straight to `END`\n"
        "- `tools` executes the real calculator and loops back to `agent` with the result "
        "appended to `state[\"messages\"]`\n"
        "- The agent can loop more than once per turn if it needs multiple calculations"
    )
