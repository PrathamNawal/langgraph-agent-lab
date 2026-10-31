"""Renders a compiled LangGraph graph as an interactive node/edge canvas
(React Flow, via streamlit-flow-component) instead of a static ASCII diagram.

Phase 1 (current): read-only - proves the topology→canvas pipeline is fully
generic (zero module-specific code in this file) before any edit behavior is
built on top of it. `mode="edit"` is accepted now and wired for node-click
selection so Phase 2 (a side panel driven by each node's own `configurable`
list) can be added without changing any call site.
"""

from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.layouts import ManualLayout
from streamlit_flow.state import StreamlitFlowState

from .topology import build_elements, layered_positions


def _to_flow_elements(nodes, edges):
    positions = layered_positions(nodes, edges)
    flow_nodes = []
    for n in nodes:
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
                "border": f"2px solid {n['style']['border']}",
                "borderRadius": "10px",
                "fontSize": "13px",
                "width": "140px",
                "textAlign": "center",
            },
        ))

    flow_edges = []
    for e in edges:
        flow_edges.append(StreamlitFlowEdge(
            id=e["id"],
            source=e["source"],
            target=e["target"],
            animated=e["conditional"],
            edge_type="smoothstep",
            style={"stroke": "#9a5b12" if e["conditional"] else "#6b6859"},
        ))

    return flow_nodes, flow_edges


def render_graph_canvas(compiled_graph, key, mode="readonly", height=430):
    """mode: "readonly" (no interaction beyond pan/zoom) or "edit" (nodes are
    clickable; returns the clicked node's id so the caller can show that
    node's `configurable` fields - the fields themselves come from
    topology.build_elements(), this function never hardcodes what any node
    kind can edit).

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

    if mode != "edit" or result_state is None or not result_state.selected_id:
        return None
    return next((n for n in nodes if n["id"] == result_state.selected_id), None)
