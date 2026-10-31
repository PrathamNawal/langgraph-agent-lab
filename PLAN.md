# LangGraph Agent Lab — Learning + Management Plan

Two things live in this repo, built together, not sequentially:

1. **The LangGraph curriculum** — a progression of increasingly complex agents, from a
   hello-world chat loop to a multi-agent production system (unchanged core sequence
   from the original plan, modules 0–8 below).
2. **The PM control panel** — a Streamlit dashboard, built once starting at Module 2 and
   *extended* at every later module, that lets you: edit an agent's configurable levers
   (prompt, model, temperature, memory strategy, tool set, routing rules) and deploy a
   new version in one click; compare every version's real eval score against every
   other version (MLflow-backed, never a mocked number); watch a live trace of the
   agent's tool calls and reasoning as it runs; and page through conversation/run
   history for monitoring.

The point of building these together: you don't just learn LangGraph's primitives, you
learn *which of those primitives are actually the levers a PM would tune in production*
(system prompt, model/temperature, memory window, tool roster, routing thresholds,
human-approval gates) and you build the muscle of measuring whether a change you made
actually helped, with a real before/after number every time.

**No dummy data, anywhere.** Every chart, every score, every "before → after" delta in
the dashboard is computed from a real model call or a real eval run against this repo's
own labeled examples — never a slider that doesn't change downstream behavior, never a
hardcoded placeholder metric. If something can't be made real yet, the UI says so
explicitly instead of faking it.

---

## Repo structure

```
langgraph-agent-lab/
  PLAN.md                 <- this file
  README.md
  requirements.txt
  .env.example
  modules/
    00_hello_world/
    01_single_tool_agent/
    02_multi_tool_memory/
    03_branching_router/
    04_human_in_the_loop/
    05_streaming/
    06_multi_agent/
    07_evals_guardrails/
    08_capstone/
  dashboard/              <- PM control panel, scaffolded starting Module 2
    app.py                <- guided nav: Overview / Try It / Configure / Monitor / Compare Versions
    config_store.py        <- versioned agent config (prompt, model, temp, memory, tools, routing)
    mlflow_tracking.py      <- experiment logging + version comparison, seeded from real eval data
    live_agent.py           <- wraps whichever module's compiled graph is "active", runs it live
  eval_sets/
    module_02.jsonl        <- small real labeled example sets per module, used for scoring versions
    ...
  mlruns/                 <- MLflow local tracking store (gitignored if it grows large)
```

Each module folder contains the actual LangGraph code for that stage (its own
`graph.py` with `StateGraph`, nodes, tools) plus a short `README.md` explaining what's
new in that module and which of its parameters get exposed as dashboard levers.

---

## How the dashboard grows with the curriculum

The dashboard isn't a one-time capstone bolt-on — it's introduced early and grows a
tab/capability at a time, mirroring what each module just taught you:

| After module | Dashboard gains | New configurable levers |
|---|---|---|
| 2 (multi-tool + memory) | First dashboard version exists: Try-It chat page, Configure page, Compare-Versions page (MLflow-backed) | System prompt, model, temperature, which tools are enabled, memory window length |
| 3 (branching) | Monitor page shows which route a request took, per conversation | Routing rules / classifier prompt |
| 4 (human-in-the-loop) | Approval queue page: pending interrupts a PM can approve/edit/reject from the dashboard itself | Which actions require approval |
| 5 (streaming) | Try-It page streams tokens + live tool-call trace instead of waiting for final answer | — |
| 6 (multi-agent) | Monitor page shows an agent-handoff timeline (which sub-agent ran, in what order) | Per-sub-agent prompt/tools/model, supervisor routing logic |
| 7 (evals/guardrails) | Compare-Versions page becomes real: every version auto-scored against the module's eval set on deploy; guardrail trip log | Guardrail rules, retry/fallback policy |
| 8 (capstone) | Everything above, polished into one coherent app with a settings panel and a runbook | All of the above, in one place |

This follows the `agentic-pm-dashboard` skill's four pillars (editable config with
one-click-deploy versioning, real MLflow-backed comparison, a guided navigation
structure instead of a flat menu, and — once this ever needs to be shared with someone
else — bring-your-own-API-key for a public visitor). Here it's framed as a lab/control
panel for yourself rather than a public PLG funnel, but the structure is the same:
Overview (what does this agent do right now) → Try It (use it) → Configure (tune it,
deploy a version) → Monitor (watch it run / approve pending actions) → Compare Versions
(did my change actually help).

---

## Module 0 — Environment & Mental Model (0.5 day)

**Concept:** LangGraph = state machine for LLM apps. A `StateGraph` has nodes
(functions), edges (control flow), and a shared `State` that flows between them.

**What you'll build:** A "hello world" graph: one node that calls an LLM, no tools, no
branching. A throwaway `st.chat_input` loop just to see it respond — this is *not* the
dashboard yet.

**Key primitives:** `StateGraph`, `START`/`END`, `add_node`, `add_edge`, `.compile()`,
`.invoke()`.

**Done when:** You can explain the difference between a LangGraph node and a LangChain
chain step, and you have a working chat box talking to an LLM through a compiled graph.

---

## Module 1 — Single Tool-Using Agent (1 day)

**Concept:** The canonical ReAct-style loop: LLM decides whether to call a tool, a tool
node executes it, result goes back to the LLM, loop until it answers.

**What you'll build:** A research assistant with one tool. Graph: `agent` node ↔
`tools` node, conditional edge routing on whether the last message has a tool call.

**Key primitives:** `add_conditional_edges`, `ToolNode`, `bind_tools`, `add_messages`
reducer, `tools_condition`.

**Done when:** You can trace what's in `state["messages"]` at each node, and you've
added a second tool yourself without copying a tutorial.

---

## Module 2 — Multi-Tool Agent with Real Memory + Dashboard v1 (1.5–2 days)

**Concept:** Short-term vs. long-term memory, structured outputs, and — new — this is
where the dashboard starts.

**What you'll build:**
- Upgrade to 3–4 tools (search, calculator, local-dataset lookup, date/time).
- A real checkpointer (`MemorySaver`, then `SqliteSaver`) so conversations survive a
  restart.
- Scaffold `dashboard/` for the first time: `config_store.py` holding a versioned
  config (system prompt, model, temperature, enabled tools, memory window),
  `live_agent.py` wrapping this module's compiled graph so the dashboard can run it
  live, `mlflow_tracking.py` seeded with a small real labeled eval set
  (`eval_sets/module_02.jsonl` — 10-20 real examples with expected answers/behavior),
  and `app.py` with Overview / Try It / Configure / Compare Versions pages.
- Deploy your first config change (e.g. tweak the system prompt) from the Configure
  page and confirm Compare Versions shows a real score difference between v1 and v2.

**Key primitives:** `MemorySaver`/`SqliteSaver`, `thread_id`, `with_structured_output`,
MLflow `log_param`/`log_metric`/`log_artifact`.

**Done when:** You can restart the app and resume a thread with memory intact, *and*
you've deployed two versions of the agent's config from the dashboard and can see which
one scored higher on the real eval set, with a trace of why.

---

## Module 3 — Branching & Explicit Control Flow (1–1.5 days)

**Concept:** Deterministic routing instead of a free-form ReAct loop: classify, then
send down one of several fixed paths.

**What you'll build:** A support-ticket triage agent: classifier node routes to one of 3
specialist nodes, converging to a respond node. Dashboard Monitor page adds a "which
route did this request take" view per logged conversation.

**Key primitives:** Multiple conditional edges, subgraphs-as-nodes, `Command` object.

**Done when:** You can add a 4th route without touching the other three nodes' code,
and the dashboard shows you the routing decision for any past conversation.

---

## Module 4 — Human-in-the-Loop & Durable Execution (1.5–2 days)

**Concept:** Pause before a side-effecting action for human approval; resume even after
a process restart.

**What you'll build:** `interrupt()` before any outbound action. Dashboard gains an
**Approval Queue** page: pending interrupts listed with Approve/Edit/Reject buttons that
resume the graph with your decision — this is a PM-facing control, not just a
developer console.

**Key primitives:** `interrupt()`, `Command(resume=...)`, Postgres/Sqlite checkpointer,
time-travel/replay from a checkpoint.

**Done when:** You can kill the process mid-approval, restart, and the paused graph (and
the dashboard's Approval Queue) still shows it waiting, state intact.

---

## Module 5 — Streaming Done Right (1 day)

**Concept:** Stream tokens, tool-call events, and intermediate state — not just the
final answer. Different `stream_mode`s matter for different UX.

**What you'll build:** Rebuild the dashboard's Try It page to stream token-by-token
output and surface intermediate events ("Calling search tool...") live via `st.status`
/ `st.write_stream`.

**Key primitives:** `.stream()`/`.astream()`, `stream_mode`, `astream_events`, custom
event emission from inside a node.

**Done when:** Watching the dashboard, you see the agent "thinking" before the final
answer streams in.

---

## Module 6 — Multi-Agent Systems (2–2.5 days)

**Concept:** Supervisor (hub-and-spoke), hierarchical teams, and swarm/handoff patterns
for splitting an unwieldy single agent into coordinated specialists.

**What you'll build:** A supervisor routing to 2-3 sub-agents (e.g. research, budgeting,
writer), each with its own tools/prompt/model. Dashboard Monitor page gains an
agent-handoff timeline. Configure page now has per-sub-agent config sections.

**Key primitives:** Subgraphs as compiled graphs used as nodes, `Command(goto=...,
graph=Command.PARENT)`, shared vs. private state schemas, `langgraph-supervisor` /
`langgraph-swarm` prebuilts (and knowing when to roll your own instead).

**Done when:** You can explain with a concrete trace why you chose supervisor vs. swarm
for this task, and the dashboard's timeline makes the handoffs legible to someone who
didn't build it.

---

## Module 7 — Reliability: Evals, Guardrails, and Error Handling (1.5–2 days)

**Concept:** Fail safely, measure whether a change helped. This is where Compare
Versions becomes a real, trustworthy tool rather than a nice-to-have.

**What you'll build:** Retry/fallback on node failure; input/output guardrails
(refuse out-of-scope requests, validate structured output and retry on mismatch); every
new config version auto-scored against the module's eval set on deploy
(LLM-as-judge or exact-match where possible); a guardrail-trip log surfaced in Monitor.

**Key primitives:** Node-level try/except routing to a fallback node, Pydantic
validation + retry loop, MLflow experiment comparison across all versions ever
deployed.

**Done when:** You have a baseline score, make one change, and the dashboard shows the
score moved — with a trace proving why, not just a number.

---

## Module 8 — Capstone: Production-Grade Agent Product (4–6 days)

**Concept:** Integrate everything into one coherent product, built the way you'd ship
it.

**What you'll build:** Pick one real-ish use case (customer-support triage agent,
research/report-writing multi-agent, or internal ops copilot) with triage branching,
tool use with approval gates, streaming, persistent memory, and a real eval suite — all
driven through the now-complete dashboard: Overview, Try It, Configure (one-click
deploy), Monitor (approval queue + handoff timeline + guardrail log), Compare Versions
(full MLflow history). Add a settings panel (model choice per sub-agent, API keys via
env vars, not hardcoded) and a short runbook (what breaks, how to debug a stuck thread,
how to roll back a bad config version).

**Done when:** You could hand this repo + README to another engineer and they could run
it, understand the graph by reading it, and tune/extend it through the dashboard
without asking you questions.

---

## Suggested pace

| Module | Focus | Est. time |
|---|---|---|
| 0 | Setup & mental model | 0.5 day |
| 1 | Single tool-using agent | 1 day |
| 2 | Multi-tool + memory + **dashboard v1** | 1.5–2 days |
| 3 | Branching / routing | 1–1.5 days |
| 4 | Human-in-the-loop | 1.5–2 days |
| 5 | Streaming | 1 day |
| 6 | Multi-agent systems | 2–2.5 days |
| 7 | Evals & guardrails | 1.5–2 days |
| 8 | Capstone | 4–6 days |

Total: roughly 3.5–4 weeks of part-time, hands-on work.

## Tracking progress

Per module, note in `modules/<module>/README.md`: what you built, one thing that
surprised you about LangGraph's behavior, one design decision you made differently from
the suggested approach and why, and (from Module 2 on) the MLflow run ID of the version
you deployed.
