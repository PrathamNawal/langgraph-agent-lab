"""Persisted agent-config version history for the Module 2 multi-tool agent.

Each "deploy" appends a new version and moves the active pointer - real
persistence to disk (JSON), not a session-only variable, so config survives
across dashboard restarts. Versions are never overwritten, which is what
makes before/after comparison on Track Performance meaningful.

Stored separately from any Module 1 config history (different file) since
this is a different agent with a different schema - mixing the two would
make "compare versions" misleading.
"""

import json
import os
from datetime import datetime, timezone

STORE_PATH = os.path.join(os.path.dirname(__file__), "data", "agent_configs_module02.json")

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful assistant with four tools: calculator, lookup_fact, "
    "days_between, and web_search. For ANY arithmetic, call calculator. For ANY "
    "question about this project (LangGraph, its modules, its dashboard, its "
    "tools), call lookup_fact. For ANY date-difference question, call days_between. "
    "For anything requiring current/live information, call web_search. Never guess "
    "at something a tool could answer exactly - you are not reliable at any of "
    "these by yourself."
)

ALL_TOOLS = ["calculator", "lookup_fact", "days_between", "web_search"]

DEFAULT_CONFIG = {
    "version": "v0",
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "temperature": 0.0,
    "model": "openai/gpt-oss-120b",
    "enabled_tools": list(ALL_TOOLS),
    "memory_window": None,
    "reason": "Initial baseline - matches Module 2's shipped config, unlimited memory window.",
    "deployed_at": None,
}


def _load_raw():
    if not os.path.exists(STORE_PATH):
        return {"versions": [DEFAULT_CONFIG], "active": "v0"}
    with open(STORE_PATH) as f:
        return json.load(f)


def _save_raw(data):
    os.makedirs(os.path.dirname(STORE_PATH), exist_ok=True)
    with open(STORE_PATH, "w") as f:
        json.dump(data, f, indent=2)


def get_active_config():
    data = _load_raw()
    active_version = data["active"]
    for v in data["versions"]:
        if v["version"] == active_version:
            return v
    return data["versions"][-1]


def list_versions():
    return _load_raw()["versions"]


def _summarize_diff(old, new):
    """Auto-fill a deploy reason from what actually changed, if the user left
    it blank - this is what makes version history read as a real changelog
    instead of a list of unlabeled timestamps."""
    changes = []
    for key, label in [
        ("system_prompt", "system prompt"), ("temperature", "temperature"), ("model", "model"),
        ("enabled_tools", "enabled tools"), ("memory_window", "memory window"),
    ]:
        if old.get(key) != new.get(key):
            changes.append("system prompt edited" if key == "system_prompt" else f"{label} {old.get(key)} -> {new.get(key)}")
    return "; ".join(changes) if changes else "No parameter changes."


def deploy_new_version(system_prompt, temperature, model, enabled_tools, memory_window, reason=""):
    data = _load_raw()
    current = get_active_config()

    next_num = len(data["versions"])
    new_version = {
        "version": f"v{next_num}",
        "system_prompt": system_prompt,
        "temperature": temperature,
        "model": model,
        "enabled_tools": enabled_tools,
        "memory_window": memory_window,
        "reason": reason.strip() or _summarize_diff(current, {
            "system_prompt": system_prompt, "temperature": temperature, "model": model,
            "enabled_tools": enabled_tools, "memory_window": memory_window,
        }),
        "deployed_at": datetime.now(timezone.utc).isoformat(),
    }

    data["versions"].append(new_version)
    data["active"] = new_version["version"]
    _save_raw(data)
    return new_version


def rollback_to(version):
    data = _load_raw()
    if not any(v["version"] == version for v in data["versions"]):
        raise ValueError(f"Unknown version: {version}")
    data["active"] = version
    _save_raw(data)
