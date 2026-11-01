"""Renders a compiled LangGraph graph as an interactive node/edge canvas
(React Flow, via streamlit-flow-component) instead of a static ASCII diagram.

Phase 1 (current): read-only - proves the topology→canvas pipeline is fully
generic (zero module-specific code in this file) before any edit behavior is
built on top of it. `mode="edit"` is accepted now and wired for node-click
selection so Phase 2 (a side panel driven by each node's own `configurable`
list) can be added without changing any call site.
"""

import streamlit as st
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.layouts import ManualLayout
from streamlit_flow.state import StreamlitFlowState

from .topology import (
    build_elements,
    layered_positions,
    expand_tool_children,
    describe_state_schema,
    describe_checkpointer,
    ROUTER_CAPTIONS,
)

# A sentinel, not None, so render_graph_canvas can tell "the caller didn't
# pass checkpointer at all" (don't show the memory badge) apart from "the
# caller explicitly passed checkpointer=None" (show the badge, as "no
# memory" - itself real information for a module like Module 1 that
# genuinely compiles without one).
_UNSET = object()

BASE_PATTERN_NOTE = (
    "This agent ↔ tool ↔ conditional-edge loop is the base pattern every later "
    "module in this curriculum extends - not a one-off diagram."
)


def _to_flow_elements(nodes, edges):
    positions = layered_positions(nodes, edges)
    nodes, edges, positions = expand_tool_children(nodes, edges, positions)

    flow_nodes = []
    for n in nodes:
        is_synthetic = n.get("synthetic", False)
        content = f"{n['style']['icon']} **{n['label']}**"
        flow_nodes.append(StreamlitFlowNode(
            id=n["id"],
            pos=positions.get(n["id"], (0, 0)),
            data={"content": content},
            node_type="input" if n["id"] == "__start__" else "output" if n["id"] == "__end__" else "default",
            # Fixed width/height, not auto-sized to content - labels are kept
            # short for exactly this reason (see topology._node_label), so the
            # layout engine can space nodes predictably without overlap.
            style={
                "background": n["style"]["bg"],
                "border": f"{'1px dashed' if is_synthetic else '2px solid'} {n['style']['border']}",
                "borderRadius": "10px" if not is_synthetic else "8px",
                "fontSize": "13px" if not is_synthetic else "11px",
                "width": "140px" if not is_synthetic else "110px",
                "textAlign": "center",
            },
        ))

    flow_edges = []
    for e in edges:
        synthetic = e.get("synthetic", False)
        flow_edges.append(StreamlitFlowEdge(
            id=e["id"],
            source=e["source"],
            target=e["target"],
            animated=e["conditional"],
            edge_type="smoothstep",
            # Synthetic tool-router -> tool edges are deliberately thin, gray
            # and non-animated - they represent "this tool is bound", not a
            # transition the graph's execution ever actually takes.
            style={"stroke": "#c8c5ba" if synthetic else ("#9a5b12" if e["conditional"] else "#6b6859")},
        ))

    return flow_nodes, flow_edges


def _render_router_captions(nodes):
    kinds_present = {n["kind"] for n in nodes if not n.get("synthetic")}
    captions = [ROUTER_CAPTIONS[k] for k in kinds_present if k in ROUTER_CAPTIONS]
    if captions:
        for c in captions:
            st.caption(f"\U0001f500 {c}")


def _render_state_and_memory(state_schema, checkpointer):
    fields = describe_state_schema(state_schema)
    show_memory = checkpointer is not _UNSET
    if not fields and not show_memory:
        return

    col_state, col_memory = st.columns(2)
    with col_state:
        if fields:
            st.markdown("**State** (`state[...]`, passed between every node)")
            for f in fields:
                reducer_note = f" — merged via `{f['reducer']}`" if f["reducer"] else " — overwritten each step"
                st.caption(f"`{f['field']}`: {f['type']}{reducer_note}")
    with col_memory:
        if show_memory:
            info = describe_checkpointer(checkpointer)
            st.markdown("**Memory** (outside the graph itself)")
            if info["present"]:
                st.caption(f"\U0001f4be `{info['type_name']}` - state for this thread_id persists across runs.")
            else:
                st.caption("⚠️ No checkpointer - state resets every run; nothing survives a restart.")


def render_graph_canvas(
    compiled_graph,
    key,
    mode="readonly",
    height=520,
    state_schema=None,
    checkpointer=_UNSET,
):
    """mode: "readonly" (no interaction beyond pan/zoom) or "edit" (nodes are
    clickable; returns the clicked node's id so the caller can show that
    node's `configurable` fields - the fields themselves come from
    topology.build_elements(), this function never hardcodes what any node
    kind can edit).

    state_schema: the module's own State TypedDict class (e.g. `State` from
    its graph.py) - introspected generically via describe_state_schema(),
    never module-specific. Omit to skip the state panel.

    checkpointer: the real checkpointer instance a module compiled its graph
    with, or None if it compiled without one - pass explicitly (not omitted)
    whenever a module wants the memory badge shown; omitting it entirely
    skips the badge for modules where memory isn't yet a concept.

    Returns: the clicked node's dict from topology.build_elements() nodes, or
    None if nothing was clicked (always None in "readonly" mode).
    """
    nodes, edges = build_elements(compiled_graph)
    flow_nodes, flow_edges = _to_flow_elements(nodes, edges)
    state = StreamlitFlowState(flow_nodes, flow_edges)

    result_state = streamlit_flow(
        key,
        state,
        height=height,
        layout=ManualLayout(),
        fit_view=True,
        show_controls=True,
        get_node_on_click=(mode == "edit"),
        pan_on_drag=True,
        allow_zoom=True,
        hide_watermark=True,
    )

    _render_router_captions(nodes)
    _render_state_and_memory(state_schema, checkpointer)
    st.caption(BASE_PATTERN_NOTE)

    if mode != "edit" or result_state is None or not result_state.selected_id:
        return None
    return next((n for n in nodes if n["id"] == result_state.selected_id), None)
