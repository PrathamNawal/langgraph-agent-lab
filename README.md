# LangGraph Agent Lab

A hands-on curriculum for building production-grade AI agents with
[LangGraph](https://github.com/langchain-ai/langgraph), paired with a self-serve
Streamlit PM control panel for configuring, monitoring, and evaluating them.

See [PLAN.md](PLAN.md) for the full module-by-module curriculum and the dashboard's
build sequence. Start at `modules/00_hello_world/`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your API key(s)
```

## Running a module's agent standalone

```bash
python modules/<module>/graph.py
```

## Running a module's app

Each module that has reached the PM-dashboard stage ships ONE Streamlit app -
a free-form Chat page plus the guided PM control panel (What is this / How it
works / Try it - Human / Try it - Agent / Play / Track Performance) - not two
separate apps:

```bash
streamlit run modules/01_single_tool_agent/app.py
streamlit run modules/02_multi_tool_memory/app.py
```

## Repo layout

```
modules/        one folder per curriculum stage, each with its own graph.py + README
modules/<n>/app.py        that module's combined Chat + PM dashboard app
modules/<n>/dashboard/    that module's config_store / mlflow_tracking / live_agent
eval_sets/      small real labeled example sets used to score agent config versions
```
