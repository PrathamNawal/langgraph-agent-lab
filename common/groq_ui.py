"""Shared Streamlit sidebar widget for entering a Groq API key and picking a model.

Used by every module's app.py from Module 0 onward so the key-handling and
live-model-lookup logic (see Module 0's README for why it's live, not hardcoded)
lives in one place instead of being copy-pasted per module.
"""

import streamlit as st


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


def groq_sidebar(default_model: str = "llama-3.1-8b-instant") -> tuple[str | None, str | None]:
    """Render the Groq API key + model sidebar widgets. Returns (api_key, model)."""
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

            default_idx = available_models.index(default_model) if default_model in available_models else 0
            model = st.selectbox("Model", available_models, index=default_idx)

    return api_key, model
