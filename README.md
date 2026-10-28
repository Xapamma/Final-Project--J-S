# Transaction Categorization Dataset

STAT 486 Fall 2026 project – Savanna & Jillian

This project builds on the import approach from [Streamlit-Budget-Dashboard](https://github.com/Xapamma/Streamlit-Budget-Dashboard), but replaces its rule-based categorization.

**Route:** C – Build a dataset and use it in a reproducible ML pipeline  
**Project plan:** [docs/project-plan.md](docs/project-plan.md)

## Data and privacy

The data comes from our personal bank statement exports. Raw statements and labeled transactions are private and are not committed to this repository. We will add a small made-up sample in the same format so others can run the pipeline, along with instructions for using your own bank exports.

## Setup

The project uses a `uv` environment (`pyproject.toml`, `uv.lock`, Python version in `.python-version`).

```bash
uv sync
```

Then follow [SETUP.md](SETUP.md): build the clean dataset headlessly with
`uv run python -m ml.bank_statment_processor`, or start the labeling app with
`uv run streamlit run dashboard.py`.

## Repository layout

| Path | Contents |
|------|----------|
| `dashboard.py`, `app.py`, `pages/` | Streamlit labeling/dashboard app |
| `dashboard/` | Shared app modules: import, export/validation, analytics, description cleaning, search, budget export, bank profiles |
| `ml/` | Headless dataset pipeline (`bank_statment_processor.py`) + ML experiments (`ml_models.py`) |
| `docs/` | `project-plan.md`, `report.md`, `CATEGORY_GUIDE.md`, `AI_USAGE.md` |
| `tests/` | Unit tests |
| `.data/` | Private raw bank CSVs (gitignored) |

## Project log

### 2026-10-27 – Project plan

Route C: We are building a cleaned dataset of both teammates' bank transactions across multiple banks.

ML question: After cleaning and combining our personal bank statements into one dataset, which machine learning techniques most accurately categorize the transactions into main categories and subcategories? And can a model trained on one person’s bank statements correctly categorize the statements of another?

Raw data stays private; a made-up sample will support reproduction. Backup scope: main categories only, a shorter date range, or one person's data.

Full plan: [docs/project-plan.md](docs/project-plan.md)
