# Module 2 — multi-tool agent with real memory

**Status:** done - agent, memory, and the merged PM dashboard are all built and verified.

## What's here

- `graph.py` — same ReAct loop shape as Module 1 (`agent` <-> `tools`, routed by
  `tools_condition`), now with four tools and a real checkpointer:
  - `calculator` — same as Module 1 (`numexpr`).
  - `lookup_fact` — keyword-match lookup over `data/local_facts.json`, a small
    real dataset of facts about this project itself (provider, modules, tools).
    This is the "local-dataset lookup" tool from PLAN.md - deliberately simple
    keyword overlap, not embeddings, so the retrieval logic stays inspectable.
  - `days_between` — date arithmetic (`YYYY-MM-DD` in, day count out).
  - `web_search` — Tavily-backed, bring-your-own `TAVILY_API_KEY`. Degrades to a
    clear error string (never a crash) if no key is set or the package isn't
    installed - not part of `eval_sets/module_02.jsonl` since its results
    aren't deterministic/offline-verifiable.
  - `get_checkpointer()` builds one shared `SqliteSaver` over a real file
    (`data/checkpoints.db`), not `MemorySaver` - `MemorySaver` keeps state only
    in process memory and would lose everything on restart, which defeats the
    point of this module.
- `app.py` — ONE deployed app, not two: a "Chat" nav page with the real memory
  behavior, plus the full PM control panel (What is this / How it works /
  Try it - Human / Try it - Agent / Play / Track Performance) as the other six
  pages. Chat only ever sends the *new* user message into `.invoke()`; the
  checkpointer merges it with whatever is already persisted for the page's
  **Conversation ID** (`thread_id`). History shown on screen is read back with
  `app.get_state(config)`, not from `st.session_state` - so a full process
  restart, not just a page refresh, still resumes the same conversation.
- `dashboard/` — `config_store.py` (versioned config incl. `enabled_tools` and
  `memory_window`), `mlflow_tracking.py`, `live_agent.py` (wraps the real
  `build_graph()`, giving every dashboard investigation its own fresh
  `thread_id` so eval runs never touch a real conversation). Imported by
  `app.py`, not run standalone.
- `data/local_facts.json` — the dataset `lookup_fact` searches.
- `eval_sets/module_02.jsonl` (project root) — 12 real, deterministic questions
  (calculator + lookup_fact + days_between) with exact-match ground truth.

## Run it

```bash
streamlit run modules/02_multi_tool_memory/app.py
```

Try asking something, then **restart the process entirely** (Ctrl-C, run the
command again) and re-enter the same Conversation ID in the sidebar - the
conversation should still be there. That's the actual thing this module is
teaching; a page refresh alone doesn't prove it, since Streamlit can appear to
"remember" things via its own session state even with no real checkpointer.

## Notes / what surprised me

- With a checkpointer attached, `.invoke()`'s *return value* is the full
  merged state, not just the new node's output - confirmed this directly with
  a throwaway two-process test before wiring up the UI: process 1 wrote one
  turn, a completely separate `python3` invocation of process 2 read the full
  history back via `app.get_state()` using only the shared SQLite file and a
  `thread_id`. That's the whole mechanism - no in-memory anything carries it.
- This simplifies `app.py` relative to Module 1 in a real way: Module 1 had to
  resend `st.session_state.messages` (the entire history) on every turn, by
  hand, because nothing else remembered it. Here, sending the new message
  alone is correct and *required* - resending old messages too would just
  double them up in the persisted state.
- `enabled_tools` on `build_graph()` is a new lever Module 1 never needed
  (there was only one tool to enable or disable). Unchecking a tool in the
  sidebar rebuilds the graph with fewer tools actually bound via `bind_tools`
  - the model genuinely cannot call a disabled tool, this isn't just hiding it
    from the UI.
- `web_search` is real but intentionally excluded from the eval set - its
  answers depend on live web content, so there's no fixed ground truth to
  score exact-match against the way there is for arithmetic, a fact lookup,
  or a date calculation.

- The top-level `dashboard/` folder was removed - each module now owns its
  own `dashboard/` subfolder and its own `app.py` that merges a Chat page with
  the full PM funnel, so a visitor to the deployed app never has to find a
  separate URL for the dashboard half.
