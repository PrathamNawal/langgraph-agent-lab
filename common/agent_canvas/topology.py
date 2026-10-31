"""Converts a compiled LangGraph graph into streamlit-flow elements.

The single rule this module exists to enforce: topology and node styling are
ALWAYS derived from the real compiled graph (`app.get_graph(xray=True)`),
never hand-drawn. A module declares what each of its nodes IS by tagging it
with `metadata={"kind": ..., "configurable": [...]}` on `add_node()` - a
built-in LangGraph feature, not a convention this project invented - and this
file is the only place that knows how to turn that into a picture. Every
module from here through the capstone just tags its nodes; it never needs to
touch this file.

KIND_STYLES is deliberately pre-populated past what Modules 1-2 use today
(router, human_approval, guardrail, subgraph) so that adding a new node kind
in a later module is a one-line addition here, not a redesign.
"""

KIND_STYLES = {
    "agent": {"icon": "\U0001f9e0", "bg": "#dce9f5", "border": "#2f5d8a"},
    "tool_router": {"icon": "\U0001f527", "bg": "#f6e3c6", "border": "#9a5b12"},
    # "tool" is a synthetic satellite node (see expand_tool_children) - one
    # per tool actually bound to a tool_router, pulled from the real graph's
    # own metadata, not a real LangGraph node. Styled visibly lighter than
    # tool_router so it never gets confused with real graph structure.
    "tool": {"icon": "\U0001fa9b", "bg": "#fbf3e5", "border": "#c89a4a"},
    "router": {"icon": "\U0001f500", "bg": "#e9e1f5", "border": "#6b4b9a"},
    "human_approval": {"icon": "\U0001f9d1", "bg": "#f5e6c6", "border": "#9a7612"},
    "guardrail": {"icon": "\U0001f6e1", "bg": "#f6dcd8", "border": "#a3372f"},
    "subgraph": {"icon": "\U0001f4e6", "bg": "#d8e8e0", "border": "#2f6a5a"},
    "terminal": {"icon": "●", "bg": "#dcefe2", "border": "#2f7a4f"},
    "default": {"icon": "⚙", "bg": "#ebe9e2", "border": "#6b6859"},
}

_TERMINAL_IDS = {"__start__", "__end__"}


def infer_kind(node_id, node):
    """kind comes from the node's own metadata when the module set one;
    falls back to a couple of structural heuristics (START/END, a ToolNode's
    self-reported __tools) so an untagged graph still renders sensibly."""
    if node_id in _TERMINAL_IDS:
        return "terminal"
    meta = node.metadata or {}
    if "kind" in meta:
        return meta["kind"]
    if "__tools" in meta:
        return "tool_router"
    return "default"


def _node_label(node_id, kind, node):
    """Deliberately short and fixed-length, never dynamic-length content like a
    tool list - a node's rendered size has to stay predictable for the layout
    engine to space nodes without overlap, which matters more as later modules
    add nodes with naturally longer content (guardrail messages, routing
    conditions). Put variable-length detail in a caption beside the canvas
    instead, the way Module 1/2's tool captions already do."""
    if node_id == "__start__":
        return "START"
    if node_id == "__end__":
        return "END"
    return node_id


def build_elements(compiled_graph):
    """Returns (nodes, edges) as plain dicts: {id, label, kind, style, configurable}
    for nodes and {id, source, target, conditional} for edges. Kept framework-
    agnostic here; canvas.py adapts this into streamlit-flow's own types, so a
    future swap of the rendering library only touches canvas.py."""
    g = compiled_graph.get_graph(xray=True)

    nodes = []
    for node_id, node in g.nodes.items():
        kind = infer_kind(node_id, node)
        style = KIND_STYLES.get(kind, KIND_STYLES["default"])
        meta = node.metadata or {}
        nodes.append({
            "id": node_id,
            "label": _node_label(node_id, kind, node),
            "kind": kind,
            "style": style,
            "configurable": meta.get("configurable", []),
            "tools": meta.get("__tools", []),
            "is_terminal": node_id in _TERMINAL_IDS,
        })

    edges = []
    for e in g.edges:
        edges.append({
            "id": f"{e.source}__{e.target}",
            "source": e.source,
            "target": e.target,
            "conditional": bool(e.conditional),
        })

    return nodes, edges


def layered_positions(nodes, edges, start_id="__start__", x_gap=170, y_gap=90):
    """Deterministic top-to-bottom positions, computed here instead of relying
    on the rendering library's own auto-layout. A small library like
    streamlit-flow-component's LayeredLayout runs its layout pass async in
    the browser, and `fit_view` can fire before that pass finishes, so the
    canvas sometimes renders scrolled to one node instead of showing the
    whole graph. Pre-computing fixed, correct positions removes that failure
    mode entirely rather than tuning around it - this is cheap to keep
    correct since every graph in this curriculum is a small DAG (with at most
    the odd back-edge, e.g. tools -> agent), not something that needs a real
    layout algorithm.
    """
    by_id = {n["id"]: n for n in nodes}
    children = {n["id"]: [] for n in nodes}
    for e in edges:
        if e["source"] in children:
            children[e["source"]].append(e["target"])

    level = {start_id: 0} if start_id in by_id else {}
    order = [start_id] if start_id in by_id else []
    i = 0
    while i < len(order):
        current = order[i]
        i += 1
        for child in children.get(current, []):
            if child not in level:
                level[child] = level[current] + 1
                order.append(child)

    # anything the BFS never reached (shouldn't happen for a connected graph)
    # still gets a position instead of silently vanishing from the canvas.
    for n in nodes:
        if n["id"] not in level:
            level[n["id"]] = 0
            order.append(n["id"])

    by_level = {}
    for nid in order:
        by_level.setdefault(level[nid], []).append(nid)

    positions = {}
    for lvl, ids in by_level.items():
        width = (len(ids) - 1) * x_gap
        for slot, nid in enumerate(ids):
            x = slot * x_gap - width / 2
            y = lvl * y_gap
            positions[nid] = (x, y)

    return positions


def expand_tool_children(nodes, edges, positions, x_gap=150, y_gap=90):
    """Turns a tool_router's bound-tool list into real satellite nodes, one
    per tool, instead of cramming names as text inside the router node.

    This exists because a glance at "tools" with no detail tells a viewer
    nothing - every module's canvas looked identical regardless of what the
    agent could actually do, which defeats the point of a *visual* canvas.
    The tool names still come straight from the compiled graph's own
    metadata (the ToolNode self-reports `__tools`), so this is still real
    data, not a drawing - it's a presentation choice (structure instead of a
    label), not a departure from "never hand-draw the graph."

    These nodes/edges are NOT part of the compiled graph (LangGraph has no
    node for "a tool exists" - a tool is a Python function the tools node
    dispatches to internally), so they're marked `synthetic: True` and given
    a visibly lighter style (KIND_STYLES["tool"]) so nobody mistakes them for
    real execution steps the graph can route through.
    """
    new_nodes = list(nodes)
    new_edges = list(edges)
    new_positions = dict(positions)

    for n in nodes:
        tool_names = n.get("tools") or []
        if n["kind"] != "tool_router" or not tool_names or n["id"] not in positions:
            continue
        parent_x, parent_y = positions[n["id"]]
        width = (len(tool_names) - 1) * x_gap
        for i, tool_name in enumerate(tool_names):
            child_id = f"__tool__{n['id']}__{tool_name}"
            new_nodes.append({
                "id": child_id,
                "label": tool_name,
                "kind": "tool",
                "style": KIND_STYLES["tool"],
                "configurable": [],
                "tools": [],
                "is_terminal": False,
                "synthetic": True,
            })
            new_edges.append({
                "id": f"{n['id']}__{child_id}",
                "source": n["id"],
                "target": child_id,
                "conditional": False,
                "synthetic": True,
            })
            new_positions[child_id] = (parent_x + i * x_gap - width / 2, parent_y + y_gap)

    return new_nodes, new_edges, new_positions
