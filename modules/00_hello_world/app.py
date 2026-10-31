"""Module 0: throwaway chat UI for the hello-world graph.

This is NOT the dashboard (that starts in Module 2) - just enough Streamlit to
see the compiled graph respond to a real message.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from graph import build_graph

st.set_page_config(page_title="Module 0: hello world", page_icon=":material/smart_toy:")

st.title("Module 0: hello world")
st.caption("One LangGraph node, no tools, no branching.")

@st.cache_data(ttl=600, show_spinner=False)
def list_groq_models(key: str) -> list[str]:
    from groq import Groq

    client = Groq(api_key=key)
    models = client.models.list().data
    chat_models = [
        m.id for m in models
        if not any(x in m.id.lower() for x in ("whisper", "tts", "guard", "prompt-guard"))
    ]
    return sorted(chat_models)


with st.sidebar:
    st.subheader("Groq API key")
    api_key = st.text_input(
        "GROQ_API_KEY",
        type="password",
        label_visibility="collapsed",
        placeholder="gsk_...",
        help="Used only for this session - never stored or sent anywhere else.",
    )

    model = None
    if api_key:
        try:
            available_models = list_groq_models(api_key)
        except Exception as e:
            st.error(f"Couldn't reach Groq with this key: {e}")
            st.stop()

        if not available_models:
            st.error("This key has no chat models available on Groq.")
            st.stop()

        default_idx = available_models.index("llama-3.1-8b-instant") if "llama-3.1-8b-instant" in available_models else 0
        model = st.selectbox("Model", available_models, index=default_idx)

if not api_key:
    st.info("Enter your Groq API key in the sidebar to start chatting.", icon=":material/key:")
    st.stop()

app = build_graph(model=model, api_key=api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask the agent anything"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = app.invoke({"messages": st.session_state.messages})
        response = result["messages"][-1].content
        st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

with st.expander("What's happening under the hood"):
    st.markdown(
        "- `StateGraph` with a single `chat` node and `START -> chat -> END` edges\n"
        "- State is `{\"messages\": [...]}`, merged via LangGraph's `add_messages` reducer\n"
        "- No tools, no memory across restarts (session-only) - that's Module 1 and 2"
    )
