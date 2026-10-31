# Agent Brief — Module 1: Calculator Tool Agent

## 1. Problem Statement

People chatting with an LLM assistant routinely ask it questions that involve
arithmetic — totals, percentages, conversions, multi-step math — and a bare LLM
answers these from pattern-matched "mental math," which is wrong often enough on
multi-digit or multi-step problems that the user can't trust any numeric answer
without re-checking it by hand. The current status quo (a plain chat agent, as built
in Module 0) silently guesses, with no way for the user to tell a correct answer from
a confident-sounding wrong one.

**This agent solves:** it answers arithmetic questions by delegating the actual
computation to a real calculator tool instead of guessing, and shows the user exactly
what expression was computed and what the tool returned — so the answer is verifiably
correct, not just plausible.

## 2. User Persona

| Field | Detail |
|---|---|
| Name | Chat assistant end-user |
| Who | Anyone using the Module 1 Streamlit chat to ask questions that involve numbers |
| Context | Mid-conversation, asks a question that requires a calculation as part of (or all of) the answer |
| Tech comfort | Low to medium — expects a chatbot to "just work," has no reason to know or care that an LLM is bad at arithmetic |
| Goal | Get a numeric answer they can trust without doing the math themselves |
| Frustration | Has been burned before by a chatbot confidently stating a wrong total/calculation |

## 2a. Job-to-be-Done

> **When I** ask a chat assistant a question that involves a calculation, **I want to**
> get an answer backed by a real computation rather than a guess, **so I can** trust
> the number without re-checking it myself.

## 3. Input / Output Specification

**Inputs**

| Input | Type | Example | Required |
|---|---|---|---|
| user_message | string | "What is 847 * 23, plus 19?" | Yes |
| conversation_history | list of role/content messages | prior turns in the session | No (empty on first turn) |
| groq_api_key | string (secret, session-only) | `gsk_...` | Yes |
| model | enum (populated live from the key's Groq account) | `openai/gpt-oss-120b` | Yes |

**Outputs**

| Output | Format | Description |
|---|---|---|
| answer | string | Natural-language response incorporating the exact computed result |
| tool_call_trace | list of {expression, result} | Every calculator call made while producing this answer, shown in an expander |

## 4. Step-by-Step Workflow (Plain English)

1. User enters their Groq API key in the sidebar; the app fetches the real list of
   chat models that key can access and lets the user pick one.
2. User types a question into the chat box.
3. The app appends the message to the session's conversation history and sends the
   full history into the compiled LangGraph agent.
4. The agent node prepends a system instruction ("always use the calculator tool for
   any arithmetic, never compute it yourself") and calls the chosen Groq model with
   the calculator tool bound.
5. If the model's response requests the calculator tool, the graph routes to the
   tools node, which evaluates the expression with a safe math evaluator (`numexpr`)
   and returns the exact result.
6. The result is appended to the conversation state and control loops back to the
   agent node, which can call the tool again or produce a final answer.
7. Once the model responds without a tool call, the loop ends and the final answer
   is shown to the user.
8. The app reconstructs which tool calls happened during this turn (matching each
   tool call to its result) and displays them in a collapsible trace under the answer.
9. The user can keep chatting; history persists only for the current browser session
   (no durable memory yet — that arrives in Module 2).

## 5. Success Metrics

| Metric | Target | How to Measure |
|---|---|---|
| Arithmetic correctness | 100% of answered calculations match the tool's exact output | Compare displayed answer's number to the trace's tool result |
| Tool-invocation rate on arithmetic prompts | 100% of prompts containing a calculation trigger at least one tool call | Inspect trace presence/absence across a test set of arithmetic prompts |
| Response latency | Under ~3 seconds per turn (one Groq round trip plus one tool round trip) | Wall-clock time from submit to rendered answer |
| Setup effort | Zero config beyond pasting a Groq API key | Time-to-first-successful-answer for a new user |

## 6. Constraints & Assumptions

**Constraints**
- The calculator tool only evaluates expressions `numexpr` can parse — no symbolic
  algebra (e.g. "solve for x"), no unit conversion, no word-problem interpretation
  beyond what the LLM extracts into a numeric expression.
- Requires a Groq API key, and specifically a model on that account that supports
  tool calling — not all Groq-hosted models do (observed directly: one model returned
  `tool calling is not supported with this model`).
- Conversation memory is session-only; restarting the Streamlit process or refreshing
  loses history (no checkpointer yet).
- No retry/fallback logic and no guardrails on tool misuse (both arrive in Module 7).
- Single tool only, by design — this module exists to teach the one-tool ReAct loop
  before Module 2 adds multiple tools.

**Assumptions**
- The user supplies a valid, working Groq API key with available quota.
- The selected model reliably follows the system prompt's instruction to always call
  the tool rather than answer from memory (verified empirically for the models tested
  during this build, not guaranteed for every model on Groq).
- The browser tab stays open for the duration of a session; state is not persisted
  anywhere outside `st.session_state`.

## 7. Contra-Indicators (When NOT to Use This Agent)

| Situation | Why it's unfit | Better alternative |
|---|---|---|
| Needs current/real-time information (prices, news, live data) | This agent has no search or retrieval tool — only arithmetic | Module 2+ agent with a search tool added |
| Needs symbolic math or algebra (e.g. "solve 2x + 3 = 11") | `numexpr` evaluates expressions, it doesn't solve equations | A CAS-backed tool (e.g. SymPy) or WolframAlpha integration |
| Regulated or audit-sensitive calculations (financial, medical dosing) | No logging, monitoring, or approval gate exists yet | Wait for Module 4 (human-in-the-loop) and Module 7 (guardrails) before using for anything consequential |
| Conversation needs to persist across sessions or app restarts | Memory is session-only, lost on refresh | Module 2's checkpointer-backed memory |
| User needs confidence/uncertainty reporting on the answer | The agent states results as fact with no confidence signal | Not solved until an eval/guardrail layer (Module 7) is added |

## 8. Data Grounding & Freshness

| Dimension | Detail |
|---|---|
| Data source | None beyond the chosen LLM's trained knowledge, plus the calculator tool's deterministic computation |
| Knowledge cutoff | Depends on the selected Groq-hosted model; not tracked or surfaced to the user today |
| Grounding method | Tool-grounding for arithmetic only (no RAG, no web search) |
| Freshness risk | Low for the arithmetic itself (deterministic, not model-dependent); High for any non-arithmetic factual claim the model might make in the same answer, since nothing checks those |
| Mitigation | Scope trust to the numeric result specifically; don't treat surrounding commentary as fact-checked |
| Upgrade path | Module 2 adds more tools (including a lookup/search tool), which is the first step toward grounding non-arithmetic claims too |

## 9. Top 3 Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Model answers simple arithmetic from memory instead of calling the tool, so a wrong answer on harder numbers goes unflagged | Medium | Medium | System prompt makes tool use mandatory for all arithmetic (already applied); a future eval (Module 7) should measure tool-skip rate directly, not just trust the prompt |
| Some Groq-hosted models don't support tool calling at all, breaking the agent entirely if selected | Medium | Medium | UI catches the error and tells the user to pick another model instead of crashing; a future version could pre-filter the model dropdown to tool-capable models only |
| No persistent memory means any restart or refresh silently loses the whole conversation | High | Low–Medium | Acceptable for a learning module; explicitly resolved in Module 2 by adding a real checkpointer |

## 10. Learning Objectives (PM Lens)

- Demonstrates the core ReAct mechanic concretely: `bind_tools` + `tools_condition` +
  `ToolNode`, and what's actually in `state["messages"]` as the loop runs.
- Makes the **system prompt** tangible as a production lever, not a personality knob —
  the exact same graph went from unreliable to reliable tool use purely by changing
  the prompt, with no code change.
- Key architectural insight: this is a single bounded tool-use loop, not a
  multi-agent or open-ended autonomous system — it terminates deterministically the
  moment the model stops requesting tools, which is a meaningfully different (and
  simpler) thing to reason about than Module 6's multi-agent handoffs.
- Natural next-level upgrade: Module 2 adds multiple tools, real persistent memory,
  and the first version of the PM control panel — where "which system prompt wins" is
  no longer a one-off manual test like this one, but a measured, versioned comparison.

> **Key insight for this project:** reliable tool use is as much a prompting problem
> as a wiring problem — the graph can be built correctly and the agent can still skip
> the tool unless the system prompt makes tool use non-optional.
