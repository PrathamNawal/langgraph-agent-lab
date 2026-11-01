# agent_canvas

Renders any module's compiled LangGraph graph as an interactive node/edge
canvas, generated live from the real graph object - never hand-drawn. This is
shared code; it has no knowledge of any specific module and should stay that
way as the curriculum adds Modules 3-8.

## How a module plugs in (the whole integration surface)

Tag each node when you build the graph, using LangGraph's own built-in
`metadata` kwarg on `add_node()` - nothing custom:

```python
graph.add_node("agent", agent_node, metadata={
    "kind": "agent",                                    # see KIND_STYLES in topology.py
    "configurable": ["system_prompt", "model", "temperature"],
})
```

Then render it:

```python
from common.agent_canvas import render_graph_canvas
render_graph_canvas(
    compiled_graph, key="unique_key_per_page",
    mode="edit",                 # nodes become clickable
    state_schema=State,          # the module's own State TypedDict class
    checkpointer=cp,             # the real checkpointer instance, or None
)
```

That's it. `topology.py` reads `app.get_graph(xray=True)` - which already
expands subgraphs - and styles each node by its `kind`. No module-specific
code belongs in this package; if you find yourself writing
`if module_name == "...":` in here, the tagging convention above is the thing
to extend instead.

`state_schema` and `checkpointer` are optional and both read real objects the
module already has - a State panel and a Memory badge are derived from them
generically (via `typing.get_type_hints` and `type(checkpointer).__name__`),
never hand-written per module. Omit `state_schema` to skip the state panel;
omit `checkpointer` entirely to skip the memory badge (pass it explicitly,
including `None`, whenever a module wants that badge shown - `None` itself
renders as "no checkpointer", which is real information for a module like
Module 1 that compiles without one).

A kind-level `ROUTER_CAPTIONS` entry in `topology.py` captions *why* a router
kind routes the way it does (e.g. "the model's own output decides which
branch runs next"). Add one entry per new router-ish kind a later module
introduces - never per module.

In `mode="edit"`, `render_graph_canvas` returns the clicked node's dict
(including its real `configurable` list from the compiled graph's own
metadata) - a module's own page is responsible for turning that into
something useful (Modules 1/2's "How it works" pages show the node's current
config values and a button to its own Play page), since that's inherently
tied to the module's own `config_store`.

## Status: Phase 1 (read-only) + a first Phase 2 slice - done

Proven on Modules 1 and 2's "How it works" page: `mode="edit"` returns the
clicked node's dict, and each module's own page renders that node's current
config values plus a link to its Play page. A full inline edit-and-deploy
form (writing straight through `config_store.deploy_new_version()` from
inside the canvas click) is still open - see below.

## Extending for later modules

- **New node kind** (Module 3's router, Module 4's `human_approval`, Module
  7's `guardrail`, Module 6's `subgraph`): add one entry to `KIND_STYLES` in
  `topology.py`. Tag the node with that `kind` in the module's `graph.py`.
  Nothing else changes.
- **Phase 2, remaining slice (inline edit mode)**: build a form generator
  keyed off a node's `configurable` list, writing through the module's
  existing `config_store.deploy_new_version()` directly from the click panel
  - the canvas becomes a different front-end onto the same config object,
  not a new one. Today the click panel shows current values and links out to
  the existing Play-page form instead of editing in place.
- **Phase 3 (observe mode, Modules 3/4/6/7's Monitor pages)**: overlay
  runtime status (current node, paused-for-approval, guardrail trip) onto the
  same `build_elements()` output, reusing this package rather than building a
  second graph renderer.
- **Subgraphs (Module 6)**: `get_graph(xray=True)` already flattens one level
  for free. Drill-down (click a `subgraph`-kind node to render *its own*
  canvas) is the real test of whether this design holds up - if it needs
  special-casing, the abstraction was wrong.

## Known limitation

`ManualLayout` with positions computed in `topology.layered_positions()` is
used instead of the library's own auto-layout (`LayeredLayout`) - the
library's layout pass runs async in the browser and `fit_view` can fire
before it finishes, which produced inconsistent, sometimes-overlapping
renders in testing. The manual version is a simple longest-path-from-start
leveling; it's correct for the small DAGs this curriculum produces but isn't
a general graph-layout algorithm - if a later module's graph gets wide or
deep enough that this looks cramped, that's the place to revisit, not a sign
something is broken.
