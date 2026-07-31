# 🧬 AutoWrangler: Agentic Data Science Pipeline

[![AutoWrangler CI](https://github.com/savinaysingh7/reprotrace-audit-orchestrator/actions/workflows/test.yml/badge.svg)](https://github.com/savinaysingh7/reprotrace-audit-orchestrator/actions/workflows/test.yml)

**AutoWrangler** is a constrained Multi-Agent workflow orchestrator that bridges the gap between Large Language Model reasoning and strictly typed Machine Learning (ML) data-preparation operations.

Built for my 2026 University Capstone.

## 🎯 Architecture

Standard generative AI tools struggle with data execution because `eval()` causes runtime errors, syntax hallucinations, and security flaws. AutoWrangler constrains LLMs to output rigid **Pydantic JSON schemas** while execution is handled by deterministic Pandas/NumPy functions.

```
Profiler → Planner (LLM) → Plan Reviewer (LLM) → Executor → QA Validator ⇄ Planner → Data Analyst (LLM)
```

### Agent Nodes
1. **Profiler**: Samples data distribution, computes memory/NaN statistics
2. **Planner**: LLM generates a structured cleaning plan (Pydantic-constrained)
3. **Plan Reviewer**: LLM validates the plan for logical consistency before execution
4. **Executor**: Deterministically applies 11 data operations via Pandas/NumPy
5. **QA Validator**: Checks for remaining nulls; loops back to Planner if validation fails
6. **Data Analyst**: Generates structured executive report with anomaly findings

### Supported Operations (11)
| Operation | Description |
|---|---|
| `cast_numeric` | Convert currency/string columns to numeric |
| `standardize_categorical_text` | Uppercase + trim, map NULL-like values to UNKNOWN |
| `parse_dates` | Convert date strings to datetime |
| `impute_mean_values` | Fill NaNs with column mean |
| `impute_median` | Fill NaNs with column median (better for skewed data) |
| `impute_mode` | Fill NaNs with most frequent value |
| `flag_zscore_anomalies` | Flag outliers using Z-score (threshold: 3.0 std) |
| `normalize_min_max` | Scale to [0, 1] range for ML prep |
| `drop_duplicates` | Remove duplicate rows |
| `drop_column` | Remove entire column |
| `leave_alone` | No transformation |

## 🚀 Key Features
- **LangGraph State Machine**: Cyclical graph with QA feedback loop
- **Smart Operation Ordering**: Auto-sorts operations (cast → impute → normalize → zscore)
- **Plan Review Agent**: LLM validates its own plan before execution
- **Model Agnostic**: Groq, Mistral, or Ollama (local) via `.env`
- **Interactive Plotly Dashboards**: Histograms, correlation heatmaps, missing data maps
- **Before/After Comparison**: Side-by-side dirty vs clean data view
- **Production Code Generator**: Auto-generates standalone Python pipeline scripts

## 💻 Tech Stack
* **AI Orchestration**: LangGraph, LangChain Core
* **Language Models**: Groq (Llama-3-70b), Mistral, Ollama (local)
* **Data Processing**: Pandas, NumPy
* **Visualizations**: Plotly
* **Constraints/Types**: Pydantic
* **Frontend**: Streamlit
* **Persistence**: SQLite (LangGraph checkpointing)

## 📁 Project Structure
```
autowrangler/
├── main.py              # Core LangGraph pipeline (6 agent nodes, graph compiler)
├── app.py               # CLI entry point
├── streamlit_app.py     # Interactive Streamlit dashboard (6 tabs)
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container deployment
├── .env                 # API keys (GROQ_API_KEY, MISTRAL_API_KEY)
├── data/
│   └── messy_data.csv   # Demo dataset
├── memory/              # SQLite checkpoints
├── reports/             # Generated reports
├── tests/
│   └── test_autowrangler.py
└── .github/workflows/
    └── test.yml         # CI/CD pipeline
```

## 🏁 Run It Locally
```bash
pip install -r requirements.txt
# Populate your GROQ_API_KEY inside the .env

# Streamlit UI
streamlit run streamlit_app.py

# CLI mode
python app.py                    # Uses demo dataset
python app.py path/to/data.csv   # Custom dataset
```

### Using Ollama (Local Models)
```bash
# Install Ollama: https://ollama.ai
ollama pull llama3.1
# Set in .env:
# AUTOWRANGLER_LLM_PROVIDER=ollama
# OLLAMA_MODEL=llama3.1
```

### Docker
```bash
docker build -t autowrangler .
docker run -p 8501:8501 --env-file .env autowrangler
```

## 🧪 Run Tests
```bash
python -m pytest tests/ -v
```
