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

## Running the PM control panel (from Module 2 onward)

```bash
streamlit run dashboard/app.py
```

## Repo layout

```
modules/        one folder per curriculum stage, each with its own graph.py + README
dashboard/      Streamlit PM control panel (config, monitoring, version comparison)
eval_sets/      small real labeled example sets used to score agent config versions
mlruns/         local MLflow tracking store
```
