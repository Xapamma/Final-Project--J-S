# Setup & Run Instructions

STAT 486 Fall 2026 – Savanna & Jillian
Updated: October 27, 2026

This repo has **two tools side by side**:

1. **Script pipeline** – normalizes raw bank CSVs into one clean table
   (`ml/bank_statment_processor.py` → `all_banks_final_categorized.csv`),
   plus the ML experiments that answer our project question (`ml/ml_models.py`).
2. **Streamlit dashboard** – the interactive labeling/import tool
   (`streamlit run dashboard.py`).

They read/write the same 9-column schema, so you can label in the app and train
in `ml/ml_models.py`, or use the scripts headlessly.

## Repository layout

| Path | Contents |
|------|----------|
| `dashboard.py`, `app.py`, `pages/` | The Streamlit app (Home / Process statements / Overview / Analytics / Budget) |
| `dashboard/` | The app's shared modules: import, export/validation, analytics, description cleaning, fuzzy search, budget PDF/Excel export, bank layout profiles |
| `ml/` | Headless dataset builder (`bank_statment_processor.py`) and ML experiments (`ml_models.py`) |
| `docs/` | `project-plan.md`, `report.md`, `CATEGORY_GUIDE.md`, `AI_USAGE.md` |
| `tests/` | Unit tests for the import/export pipeline |
| `.data/` | Private raw bank CSVs — **never committed** (see README privacy notes) |

## 1. Prerequisites

* [uv](https://docs.astral.sh/uv/) (the project uses a `uv` environment)
* Python 3.13 (see `.python-version`)

## 2. Install the environment

```bash
uv sync
```

This installs everything in `pyproject.toml` into the project's `.venv`.

## 3. Script pipeline (headless / reproducible run)

```bash
# Build the clean, UNLABELED dataset from the CSVs in .data/
uv run python -m ml.bank_statment_processor
```

Outputs:

* `all_banks_final_categorized.csv` – columns:
  `date, description, raw_description, type, amount, main_category,
  sub_category, bank, account`.
* `main_category` / `sub_category` start **blank on purpose**. Fill them in by
  hand (the app is the easiest way) or with the ML models once implemented.

## 4. Streamlit app (interactive labeling)

```bash
uv run streamlit run dashboard.py
```

Pages:

* **Home** – overview of the workflow.
* **Process statements** – upload bank CSVs, pick bank/column mapping, review
  rows, assign category pairs, download the labeled CSV. Categories are never
  auto-filled; use the review panel to label every row before download is enabled.
* **Overview / Analytics** – charts of income, spending, and category totals.
* **Budget** – monthly budget planning.

## 5. ML workflow (our project question)

1. Label transactions in the app (or in the CSV) — this is the training data.
2. Add the ML library: `uv add scikit-learn`.
3. Implement the TODOs in `ml/ml_models.py` (thin
   stubs + instructions are already there).
4. Train and categorize everything:

```bash
uv run python -c "from ml.ml_models import train_and_categorize; train_and_categorize()"
```

5. Cross-person experiment (after adding a `person` column A/B):

```bash
uv run python -c "from ml.ml_models import cross_person_experiment; cross_person_experiment()"
```

6. Compare against the baselines (`baseline_most_common`) — see
   `CATEGORY_GUIDE.md` and `report.md` for the category vocabulary and results.

## 6. Using your own bank exports (data instructions)

* Drop one sub-folder per bank under `.data/` (e.g. `.data/sofi/sofi_checking.csv`).
* Known layouts (Goldenwest, SoFi, Capital One) are auto-detected by filename;
  the Streamlit app also has a column-mapping UI for other banks.
* **Privacy:** raw statements and the labeled dataset are private per our plan —
  do not commit `.data/` or labeled CSVs. Only the made-up sample (to be added)
  goes into the repo for reproduction.

## 7. Tests

```bash
uv run python -m unittest discover -s tests -v
```

Run after any change that touches importing, onboarding, or export logic.

## 8. Common issues

* `streamlit` not found → run `uv sync` first.
* App rejects a download → some rows still lack a category pair; use the
  "category review" panel on the Process page to finish labeling.
* New bank CSV won't parse → use the app's "Other / new bank" flow and map the
  columns; then note the layout below `dashboard/bank_profiles.json`.