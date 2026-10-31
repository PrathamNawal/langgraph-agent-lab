# Module 1 — single tool-using agent

**Status:** done.

## What's here

- `graph.py` — the canonical ReAct loop: `agent` node (LLM bound to tools via
  `bind_tools`) <-> `tools` node (`ToolNode`), routed by `tools_condition`. Two tools:
  `calculator` (a safe arithmetic evaluator, `numexpr`) and `convert_units` (length/weight
  unit conversion). A system prompt forces the agent to always call the right tool
  rather than guess, for both arithmetic and unit conversion.
- `app.py` — ONE deployed app: a free-form "Chat" page (via `common/groq_ui.py`, shared
  with Module 0) showing the tool-call trace per answer, plus the full PM control panel
  (What is this / How it works / Try it - Human / Try it - Agent / Play / Track
  Performance) as the other six nav pages - not a separate dashboard app. "How it works"
  renders the graph as a real interactive canvas (`common/agent_canvas`), generated from
  `app.get_graph()` - see that package's README for how later modules plug into it.
- `dashboard/` — `config_store.py` (versioned config incl. `enabled_tools`),
  `mlflow_tracking.py`, `live_agent.py` (wraps the real `build_graph()` for the
  dashboard's live-investigation page). Imported by `app.py`, not run standalone.

## Run it

```bash
streamlit run modules/01_single_tool_agent/app.py
```

## Notes / what surprised me

- Without an explicit system prompt, `llama-3.1-8b-instant` answered `847*23` correctly
  from memory/computation without calling the tool at all - which *looked* fine but
  isn't something to rely on (it would silently fail on harder numbers). Adding
  "you must call the calculator tool for ANY arithmetic, you are not reliable at mental
  math" to the system prompt made tool use consistent. First real lesson in why the
  system prompt is a production lever, not just a personality knob.
- Not every Groq-hosted model supports tool calling - one that was fine for Module 0's
  plain chat threw `tool calling is not supported with this model` here. `app.py` now
  catches that and tells the user to pick another model instead of crashing; a later
  module could filter the model list to tool-capable ones only, but Groq doesn't expose
  that as metadata on `models.list()`, so for now it's a runtime check.
- Design decision: the tool-call trace is rebuilt per-turn by matching `AIMessage.tool_calls`
  ids to the corresponding `ToolMessage` in `state["messages"]` - this is the same
  message-list structure Module 2's dashboard will read from for monitoring, so it's
  worth understanding this matching logic now.
- Added `convert_units` as the second tool (PLAN.md's "done when" for this module).
  `TOOLS` and `SYSTEM_PROMPT` both needed updating - binding a second tool via `bind_tools`
  isn't enough on its own; the forcing instruction had to explicitly call out unit
  conversion too, or the model could plausibly skip the tool the same way it originally
  skipped `calculator`. The trace UI in `app.py` was hardcoded to the word "calculator"
  and to a single `expression` arg, so it had to be generalized to render any tool's
  name and keyword args - worth remembering that UI assumptions baked in for a one-tool
  agent don't automatically survive adding a second tool.
- Merged what used to be a separate `dashboard/` deployment into this module's own
  `app.py` as extra nav pages, after feedback that a stranger visiting the deployed
  agent shouldn't have to find a *different* URL to see its PM control panel. The
  dashboard's `config_store`/`mlflow_tracking`/`live_agent` files moved into
  `modules/01_single_tool_agent/dashboard/` and import the real `graph.py` one
  directory up - same pattern Module 2 follows.
