# Module 0 — hello world

**Status:** done.

## What's here

- `graph.py` — the smallest LangGraph agent: one `StateGraph`, one `chat` node, no
  tools, no branching. `START -> chat -> END`. Uses Groq (`ChatGroq`) as the LLM.
- `app.py` — a throwaway Streamlit chat UI (not the dashboard — that starts Module 2).
  The Groq API key is entered in the sidebar at runtime (never read from `.env` or
  hardcoded), and the model dropdown is populated live from `client.models.list()`
  against that key, rather than a hardcoded model name — Groq retires/renames models
  often enough that a hardcoded default 404s.

## Run it

```bash
streamlit run modules/00_hello_world/app.py
```

## Notes / what surprised me

- Groq's hosted model catalog changes: `llama-3.3-70b-versatile` and
  `llama-3.1-8b-instant` both 404'd during this build even though they're commonly
  referenced in tutorials. Fetching `models.list()` from the actual key at runtime is
  the only reliable way to know what's available — this became the first real "don't
  fake it" lesson from the dashboard-building skill, applied a module early.
- `state["messages"]` uses LangGraph's `add_messages` reducer even in a one-node graph —
  worth seeing this wiring before Module 1 adds a loop over it.
