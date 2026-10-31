"""Persisted agent-config version history for Module 1's agent.

Each "deploy" appends a new version and moves the active pointer - real
persistence to disk (JSON), not a session-only variable, so config survives
across dashboard restarts. Versions are never overwritten, which is what
makes before/after comparison on Track Performance meaningful.
"""

import json
import os
from datetime import datetime, timezone

STORE_PATH = os.path.join(os.path.dirname(__file__), "data", "agent_configs.json")

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful assistant. For ANY arithmetic, no matter how simple it looks "
    "(even single-digit math), you must call the calculator tool rather than computing "
    "it yourself - you are not reliable at mental math and must never guess. For ANY "
    "unit conversion, you must call the convert_units tool rather than recalling the "
    "conversion factor from memory - you are not reliable at that either."
)

ALL_TOOLS = ["calculator", "convert_units"]

DEFAULT_CONFIG = {
    "version": "v0",
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "temperature": 0.0,
    "model": "openai/gpt-oss-120b",
    "enabled_tools": list(ALL_TOOLS),
    "reason": "Initial baseline - matches Module 1's shipped config.",
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
        ("system_prompt", "system prompt"), ("temperature", "temperature"),
        ("model", "model"), ("enabled_tools", "enabled tools"),
    ]:
        if old.get(key) != new.get(key):
            changes.append("system prompt edited" if key == "system_prompt" else f"{label} {old.get(key)} -> {new.get(key)}")
    return "; ".join(changes) if changes else "No parameter changes."


def deploy_new_version(system_prompt, temperature, model, enabled_tools, reason=""):
    data = _load_raw()
    current = get_active_config()

    next_num = len(data["versions"])
    new_version = {
        "version": f"v{next_num}",
        "system_prompt": system_prompt,
        "temperature": temperature,
        "model": model,
        "enabled_tools": enabled_tools,
        "reason": reason.strip() or _summarize_diff(current, {
            "system_prompt": system_prompt, "temperature": temperature,
            "model": model, "enabled_tools": enabled_tools,
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
