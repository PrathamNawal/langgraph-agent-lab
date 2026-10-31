# Module 1 — single tool-using agent

**Status:** done.

## What's here

- `graph.py` — the canonical ReAct loop: `agent` node (LLM bound to one tool via
  `bind_tools`) <-> `tools` node (`ToolNode`), routed by `tools_condition`. The one
  tool is `calculator`, a safe arithmetic evaluator (`numexpr`). A system prompt
  forces the agent to always call the tool for arithmetic rather than guess.
- `app.py` — chat UI (via `common/groq_ui.py`, shared with Module 0) that also shows,
  per answer, a "Tool calls made for this answer" trace: the exact expression sent to
  the tool and the real result it returned.

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
