"""PM control panel for Module 1's calculator agent.

A guided 6-step funnel: what the agent is, how its LangGraph loop works, try
the same arithmetic yourself, watch the real agent solve it live, tweak its
prompt/model/temperature and deploy a new version, then see the measured
before/after on real eval questions - no dummy numbers anywhere.

Run with: streamlit run dashboard/app.py
"""

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import config_store  # noqa: E402
import mlflow_tracking  # noqa: E402
from live_agent import run_investigation_stream  # noqa: E402
from eval_sets.calculator_eval import EVAL_QUESTIONS  # noqa: E402
from common.groq_ui import list_groq_models  # noqa: E402

st.set_page_config(page_title="Calculator Agent — PM Dashboard", page_icon=":material/calculate:", layout="wide")

NAV_OPTIONS = [
    "1. What is this?",
    "2. How it works",
    "3. Try it — Human",
    "4. Try it — Agent",
    "5. Play — Tweak the Agent",
    "6. Track Performance",
]
STEP_LABELS = ["What", "How", "Human", "Agent", "Play", "Track"]

QUICK_EXPERIMENTS = {
    "remove_forcing": {
        "label": "🎲 Remove the tool-forcing instruction",
        "help": "Recreates the exact bug found while building Module 1: without an explicit instruction, the model sometimes answers arithmetic from memory instead of calling the tool.",
        "changes": {"system_prompt": "You are a helpful assistant."},
    },
    "stricter": {
        "label": "🔒 Make the instruction even stricter",
        "help": "Adds an explicit consequence for skipping the tool, to see if it changes tool-call reliability.",
        "changes": {"system_prompt": config_store.DEFAULT_SYSTEM_PROMPT + " If you answer without calling the tool, your answer will be treated as wrong even if it happens to be correct."},
    },
    "warmer": {
        "label": "🌡️ Raise the temperature",
        "help": "Keeps the same prompt, but makes the model less deterministic — useful for seeing whether tool-use reliability holds up.",
        "changes": {"temperature": 0.7},
    },
}


def goto(page_name, **session_updates):
    st.session_state["nav_override"] = page_name
    for k, v in session_updates.items():
        st.session_state[k] = v
    st.rerun()


def mark_step_done(step_idx):
    st.session_state.setdefault("progress", {i: False for i in range(6)})[step_idx] = True


def render_progress(current_idx):
    prog = st.session_state.setdefault("progress", {i: False for i in range(6)})
    if current_idx in (0, 1, 5):
        prog[current_idx] = True
    cols = st.columns(6)
    for i, col in enumerate(cols):
        done = prog.get(i, False)
        is_current = (i == current_idx)
        icon = "✅" if done else ("▶️" if is_current else "⚪")
        text = f"{icon} {i + 1}. {STEP_LABELS[i]}"
        col.markdown(f"**{text}**" if is_current else text)
    st.progress(sum(prog.values()) / 6)
    st.write("")


@st.dialog("👋 Welcome")
def welcome_dialog():
    st.markdown(
        "This is a guided, 6-step tour (about 5 minutes) of the calculator agent from "
        "Module 1 — built to be explored, not just read about.\n\n"
        "You'll see how its LangGraph loop works, try the same arithmetic yourself, "
        "watch the real agent solve it live, tweak its prompt or model, and measure "
        "whether your change actually helped."
    )
    if st.button("Start the tour →", type="primary"):
        st.session_state["welcomed"] = True
        st.rerun()


if "seeded" not in st.session_state:
    mlflow_tracking.seed_historical_baselines()
    st.session_state["seeded"] = True

if "nav_override" in st.session_state:
    st.session_state["nav"] = st.session_state.pop("nav_override")

if "welcomed" not in st.session_state:
    welcome_dialog()


def item_by_id(item_id):
    return next(q for q in EVAL_QUESTIONS if q["id"] == item_id)


# ---------- sidebar: nav + Groq credential ----------
st.sidebar.title("Calculator Agent")
page = st.sidebar.radio("View", NAV_OPTIONS, key="nav")

st.sidebar.divider()
st.sidebar.caption("Groq API key — only needed for step 4 (session only, never written to disk)")
api_key = st.sidebar.text_input("GROQ_API_KEY", type="password", placeholder="gsk_...", label_visibility="collapsed")

active_config = config_store.get_active_config()
st.sidebar.divider()
st.sidebar.caption(f"Active config: **{active_config['version']}** ({active_config['model']}, temp {active_config['temperature']})")

render_progress(NAV_OPTIONS.index(page))


# ============================================================
# 1. WHAT IS THIS?
# ============================================================
if page == "1. What is this?":
    st.title("Calculator Agent")
    st.markdown("#### A LangGraph agent that answers arithmetic by calling a real calculator — instead of guessing.")

    st.markdown(
        "A bare LLM answers math questions by pattern-matching a plausible-looking number, "
        "which is wrong often enough on multi-digit or multi-step problems that you can't trust "
        "it without checking by hand. This agent instead delegates the actual computation to a "
        "real tool and loops back with the exact result before answering."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Tools available", "1", help="calculator — evaluates a math expression exactly via numexpr.")
    c2.metric("Eval questions", str(len(EVAL_QUESTIONS)), help="Real arithmetic questions with hand-computed ground truth, in eval_sets/calculator_eval.py.")
    c3.metric("Config versions deployed", str(len(config_store.list_versions())), help="Every prompt/model/temperature change you deploy becomes a new, comparable version.")

    st.info(
        "The honest tension: while building this module, the SAME agent answered **847 × 23 correctly "
        "from memory** on the first try, with no tool call at all — which looked fine, until you realize "
        "nothing guarantees it'll be right on the next one. That's what \"Play\" (step 5) lets you "
        "reproduce and measure on purpose."
    )

    st.divider()
    st.subheader("What you can do here")
    st.markdown("""
1. **See how it works** — the LangGraph loop that decides when to call the tool
2. **Try it yourself** — answer a real arithmetic question by hand, scored exactly
3. **Try it as the agent** — watch the real graph run live, tool calls and all
4. **Play** — edit its system prompt, model, or temperature and deploy a new version
5. **Track performance** — see exactly how your change moved real accuracy on the eval set
""")

    if st.button("See how it works →", type="primary"):
        goto("2. How it works")


# ============================================================
# 2. HOW IT WORKS
# ============================================================
elif page == "2. How it works":
    st.title("How it works")
    st.markdown(
        "Every question goes through the same loop: the model decides whether this needs "
        "the calculator, and if so, the real tool runs and the model gets the exact result "
        "back before answering."
    )

    st.markdown("#### The agent loop")
    st.markdown("""
```
  Question comes in
        │
        ▼
  agent node (LLM + calculator bound via bind_tools)
        │
        ▼
  tool_calls present?  ──No──►  answer directly  ──►  END
        │ Yes
        ▼
  tools node runs the real calculator (numexpr)
        │
        ▼
  back to agent node, now with the exact result in state
        │
        └──► loops until the model stops requesting the tool
```
""")

    st.markdown("**🔧 calculator**")
    st.caption("Evaluates a math expression exactly, e.g. \"123*9\" → \"1107\". The one tool this agent has.")

    st.divider()
    st.caption("Full source: modules/01_single_tool_agent/graph.py")

    if st.button("Try it yourself →", type="primary"):
        goto("3. Try it — Human")


# ============================================================
# 3. TRY IT — HUMAN
# ============================================================
elif page == "3. Try it — Human":
    st.title("Try it — do the arithmetic yourself")
    st.caption("Real questions, real ground truth. Your answer is logged and scored exactly like the agent's.")

    already_judged = mlflow_tracking.judged_ids("human", active_config["version"])
    pool = [q for q in EVAL_QUESTIONS if q["id"] not in already_judged]

    if "human_item_id" not in st.session_state or st.session_state["human_item_id"] not in [q["id"] for q in pool]:
        if pool:
            st.session_state["human_item_id"] = pool[0]["id"]

    if not pool:
        st.success("You've answered every eval question under this config version!")
    else:
        pool_ids = [q["id"] for q in pool]
        col_pick, col_random = st.columns([3, 1])
        with col_pick:
            item_id = st.selectbox("Question", pool_ids,
                                    index=pool_ids.index(st.session_state["human_item_id"])
                                    if st.session_state["human_item_id"] in pool_ids else 0)
        with col_random:
            if st.button("🎲 Random question"):
                import random
                st.session_state["human_item_id"] = random.choice(pool_ids)
                st.rerun()

        item = item_by_id(item_id)
        st.info(item["question"])

        with st.form("human_judgment_form"):
            your_answer = st.number_input("Your answer", step=1.0, format="%g")
            submitted = st.form_submit_button("Submit answer")

        if submitted:
            correct = abs(your_answer - item["answer"]) < 0.01
            mlflow_tracking.log_judgment(
                judge_type="human", config_version=active_config["version"], item_id=item_id,
                correct=correct, predicted_value=your_answer, expected_value=item["answer"],
            )
            st.session_state["last_human_item_id"] = item_id
            st.session_state["last_human_correct"] = correct
            st.session_state["human_item_id"] = None
            mark_step_done(2)
            st.rerun()

        if st.session_state.get("last_human_item_id") is not None:
            iid = st.session_state["last_human_item_id"]
            correct = st.session_state["last_human_correct"]
            answer = item_by_id(iid)["answer"]
            if correct:
                st.success(f"✓ Correct — the answer is **{answer}**.")
            else:
                st.error(f"✗ Not quite — the correct answer is **{answer}**.")
            if st.button("Now see what the agent does on this same question →", type="primary"):
                iid_to_pass = st.session_state.pop("last_human_item_id")
                st.session_state.pop("last_human_correct", None)
                goto("4. Try it — Agent", handoff_item_id=iid_to_pass)


# ============================================================
# 4. TRY IT — AGENT
# ============================================================
elif page == "4. Try it — Agent":
    st.title("Try it — the real agent, live")
    st.caption(f"Running against the active config: **{active_config['version']}** — "
               f"model {active_config['model']}, temperature {active_config['temperature']}")

    if not api_key:
        st.warning(
            "**No Groq API key set.** Three steps to run a live investigation:\n\n"
            "1. Get a free key → [console.groq.com/keys](https://console.groq.com/keys) (about 30 seconds)\n"
            "2. Paste it into the sidebar field\n"
            "3. Come back here and click **Investigate live**"
        )

    already_judged = mlflow_tracking.judged_ids("agent", active_config["version"])
    pool_ids = [q["id"] for q in EVAL_QUESTIONS if q["id"] not in already_judged]
    all_ids = [q["id"] for q in EVAL_QUESTIONS]

    handoff_id = st.session_state.pop("handoff_item_id", None)
    if handoff_id is not None:
        st.session_state["agent_item_id"] = handoff_id
        st.caption(f"↪ Continuing from your human answer on question **{handoff_id}**")
    elif "agent_item_id" not in st.session_state or st.session_state["agent_item_id"] not in all_ids:
        st.session_state["agent_item_id"] = (pool_ids[0] if pool_ids else all_ids[0])

    if not pool_ids and handoff_id is None:
        st.success("The agent has answered every eval question under this config version!")
    else:
        col_pick, col_random = st.columns([3, 1])
        with col_pick:
            item_id = st.selectbox("Question", all_ids,
                                    index=all_ids.index(st.session_state["agent_item_id"]))
        with col_random:
            if st.button("🎲 Random question", key="agent_random"):
                import random
                st.session_state["agent_item_id"] = random.choice(pool_ids or all_ids)
                st.rerun()

        item = item_by_id(item_id)
        st.info(item["question"])

        if st.button("▶ Investigate live", type="primary", disabled=not api_key):
            trace_area = st.container()
            final_result = None
            n_tool_calls = 0

            with st.status("Running the agent...", expanded=True) as status:
                for event in run_investigation_stream(item, active_config, api_key):
                    if event["type"] == "error":
                        status.update(label="Run failed", state="error")
                        st.error(event["message"])
                        break
                    elif event["type"] == "turn":
                        trace_area.markdown(f"**Turn {event['turn']}** — calling the model")
                    elif event["type"] == "tool_call":
                        n_tool_calls += 1
                        trace_area.markdown(f"→ called `calculator({event['input'].get('expression', event['input'])})`")
                        trace_area.code(event["result"], language="text")
                    elif event["type"] == "final":
                        final_result = event
                        status.update(label="Done", state="complete")

            if final_result:
                st.divider()
                st.markdown(f"**Agent's answer:** {final_result['answer_text']}")
                predicted = final_result.get("predicted_value")
                correct = predicted is not None and abs(predicted - item["answer"]) < 0.01

                col_a, col_b = st.columns(2)
                col_a.metric("Tool calls made", n_tool_calls)
                col_b.metric("Turns used", final_result.get("turns_used", 0))

                if st.button("Reveal the correct answer"):
                    if correct:
                        st.success(f"✓ Matches — correct answer: **{item['answer']}**.")
                    else:
                        st.error(f"✗ Does not match — correct answer: **{item['answer']}**, agent said **{predicted}**.")

                mlflow_tracking.log_judgment(
                    judge_type="agent", config_version=active_config["version"], item_id=item_id,
                    correct=correct, turns_used=final_result.get("turns_used"), n_tool_calls=n_tool_calls,
                    model=active_config["model"], temperature=active_config["temperature"],
                    predicted_value=predicted, expected_value=item["answer"],
                )
                mark_step_done(3)

        st.divider()
        if st.button("Play — tweak how the agent behaves →"):
            goto("5. Play — Tweak the Agent")


# ============================================================
# 5. PLAY — TWEAK THE AGENT
# ============================================================
elif page == "5. Play — Tweak the Agent":
    st.title("Play — you're the PM now")
    st.caption("Change the agent's system prompt, model, or temperature. One click deploys a new version — past versions stay comparable.")

    st.markdown("**Quick experiments** — one click pre-fills the form below, review it, then deploy:")
    exp_cols = st.columns(len(QUICK_EXPERIMENTS))
    for col, (key, exp) in zip(exp_cols, QUICK_EXPERIMENTS.items()):
        with col:
            if st.button(exp["label"], help=exp["help"], key=f"exp_{key}"):
                st.session_state["config_prefill"] = exp["changes"]
                st.rerun()

    prefill = st.session_state.get("config_prefill", {})
    defaults = {**active_config, **prefill}

    available_models = []
    if api_key:
        try:
            available_models = list_groq_models(api_key)
        except Exception:
            available_models = []

    st.divider()
    with st.form("deploy_form"):
        system_prompt = st.text_area("System prompt", value=defaults["system_prompt"], height=160)
        col1, col2 = st.columns(2)
        with col1:
            if available_models:
                idx = available_models.index(defaults["model"]) if defaults["model"] in available_models else 0
                model = st.selectbox("Model", available_models, index=idx)
            else:
                model = st.text_input("Model", value=defaults["model"], help="Enter your Groq API key in the sidebar to pick from your live model list.")
        with col2:
            temperature = st.slider("Temperature", 0.0, 1.0, float(defaults["temperature"]), 0.05)

        reason = st.text_input("Reason for this change (optional — auto-filled from the diff if left blank)")
        deployed = st.form_submit_button("🚀 Deploy new version", type="primary")

    if deployed:
        new_version = config_store.deploy_new_version(
            system_prompt=system_prompt, temperature=temperature, model=model, reason=reason,
        )
        st.session_state.pop("config_prefill", None)
        mark_step_done(4)
        st.success(f"Deployed **{new_version['version']}**: {new_version['reason']}")
        if st.button("See how this changed performance →", type="primary"):
            goto("6. Track Performance")

    st.divider()
    st.subheader("Version history")
    versions = config_store.list_versions()
    hist_df = pd.DataFrame([
        {"version": v["version"], "model": v["model"], "temperature": v["temperature"],
         "reason": v["reason"], "deployed_at": v["deployed_at"]}
        for v in reversed(versions)
    ])
    st.dataframe(hist_df, width="stretch")

    rollback_target = st.selectbox("Roll back to version", [v["version"] for v in reversed(versions)])
    if st.button("Roll back"):
        config_store.rollback_to(rollback_target)
        st.rerun()


# ============================================================
# 6. TRACK PERFORMANCE
# ============================================================
elif page == "6. Track Performance":
    st.title("Track Performance")
    st.caption("Every human and agent answer, grouped by config version — logged to MLflow against exact ground truth.")

    summary = mlflow_tracking.summarize_by_group()
    if summary.empty:
        st.info("No answers logged yet. Go try it as a human or the agent first.")
    else:
        agent_summary = summary[summary["tags.judge_type"] == "agent"].copy()
        if len(agent_summary) >= 2:
            prev_row, latest_row = agent_summary.iloc[-2], agent_summary.iloc[-1]
            delta = latest_row["accuracy"] - prev_row["accuracy"]
            arrow = "📈" if delta > 0 else ("📉" if delta < 0 else "➡️")
            st.markdown(
                f"### {arrow} Your last agent config change moved accuracy from "
                f"**{prev_row['accuracy']:.0%}** (`{prev_row['tags.config_version']}`) to "
                f"**{latest_row['accuracy']:.0%}** (`{latest_row['tags.config_version']}`)"
            )
        else:
            st.markdown("Deploy at least two config versions and run the agent under each to see a before/after delta here.")

        st.divider()
        st.subheader("Performance by judge type & config version")
        display_cols = ["tags.judge_type", "tags.config_version", "n", "accuracy", "accuracy_ci", "avg_turns", "avg_tool_calls"]
        st.dataframe(summary[[c for c in display_cols if c in summary.columns]], width="stretch")

        st.subheader("Accuracy by config version")
        st.bar_chart(
            summary.pivot_table(index="tags.config_version", columns="tags.judge_type", values="accuracy"),
            width="stretch",
        )

        if not agent_summary.empty and "avg_tool_calls" in agent_summary.columns:
            st.subheader("Average tool calls per question (agent only)")
            st.bar_chart(agent_summary.set_index("tags.config_version")[["avg_tool_calls"]], width="stretch")

        with st.expander("Raw MLflow run log"):
            st.dataframe(mlflow_tracking.get_all_runs(), width="stretch")
