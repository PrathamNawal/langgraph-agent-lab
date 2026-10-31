"""MLflow-backed experiment tracking for Module 1's dashboard.

One MLflow run per judged eval question, tagged with the config version and
judge type (human/agent), so real before/after comparison across config
deploys is a group-by away - not a hand-rolled CSV someone has to remember
to update.

Uses the SQLite tracking backend, not the plain filesystem ("file:./mlruns")
backend - MLflow's filesystem store is in maintenance mode in current
versions and can throw at startup; sqlite:/// is the supported local option
and needs no server.
"""

import os

import mlflow
import pandas as pd

TRACKING_DB = os.path.join(os.path.dirname(__file__), "data", "mlflow.db")
EXPERIMENT_NAME = "langgraph-agent-lab-module01"

mlflow.set_tracking_uri(f"sqlite:///{os.path.abspath(TRACKING_DB)}")
mlflow.set_experiment(EXPERIMENT_NAME)


def log_judgment(*, judge_type, config_version, item_id, correct, turns_used=None,
                  n_tool_calls=None, model=None, temperature=None, predicted_value=None,
                  expected_value=None):
    """judge_type: 'human' or 'agent'. correct: whether the final numeric answer
    matched the question's known ground truth exactly."""
    with mlflow.start_run(run_name=f"{judge_type}_{config_version}_{item_id}"):
        mlflow.set_tag("judge_type", judge_type)
        mlflow.set_tag("config_version", config_version)
        mlflow.set_tag("item_id", str(item_id))
        if model:
            mlflow.set_tag("model", model)

        if temperature is not None:
            mlflow.log_param("temperature", temperature)
        if predicted_value is not None:
            mlflow.set_tag("predicted_value", str(predicted_value))
        if expected_value is not None:
            mlflow.set_tag("expected_value", str(expected_value))

        mlflow.log_metric("correct", int(correct))
        if turns_used is not None:
            mlflow.log_metric("turns_used", turns_used)
        if n_tool_calls is not None:
            mlflow.log_metric("n_tool_calls", n_tool_calls)


def _wilson_ci(successes, n, z=1.96):
    """Wilson score interval - meaningfully more honest than a naive
    proportion +/- stderr at the small n a dashboard's live sessions will
    actually produce (single digits to low hundreds of judgments)."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def get_all_runs():
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        return pd.DataFrame()
    return mlflow.search_runs(experiment_ids=[exp.experiment_id])


def summarize_by_group(group_cols=("tags.judge_type", "tags.config_version")):
    runs = get_all_runs()
    if runs.empty:
        return pd.DataFrame()

    rows = []
    for keys, group in runs.groupby(list(group_cols)):
        n = len(group)
        correct = group["metrics.correct"].sum()
        accuracy = correct / n
        acc_lo, acc_hi = _wilson_ci(correct, n)

        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row.update({
            "n": n,
            "accuracy": round(accuracy, 3),
            "accuracy_ci": f"{acc_lo:.0%}-{acc_hi:.0%}",
        })
        if "metrics.turns_used" in group.columns:
            row["avg_turns"] = round(group["metrics.turns_used"].mean(), 2)
        if "metrics.n_tool_calls" in group.columns:
            row["avg_tool_calls"] = round(group["metrics.n_tool_calls"].mean(), 2)
        rows.append(row)

    return pd.DataFrame(rows)


def judged_ids(judge_type, config_version):
    runs = get_all_runs()
    if runs.empty or "tags.item_id" not in runs.columns:
        return set()
    mask = (runs.get("tags.judge_type") == judge_type) & (runs.get("tags.config_version") == config_version)
    return set(runs.loc[mask, "tags.item_id"].astype(str))


def seed_historical_baselines():
    """No prior labeled/eval run exists for this project - an honest empty
    Track Performance page on first launch is correct here; fabricating a
    baseline number would violate the no-dummy-data rule."""
    pass
