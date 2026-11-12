# Setup & Run Instructions

STAT 486 Fall 2026 – Savanna & Jillian
Updated: October 27, 2026

This repo has **two tools side by side**:

1. **Script pipeline** – normalizes raw bank CSVs into one clean table
   (`ml/bank_statment_processor.py` → `savanna_transactions.csv` +
   `jillian_transactions.csv`),
   plus the ML experiments that answer our project question (`ml/ml_models.py`).
2. **Streamlit dashboard** – the interactive labeling/import tool
   (`streamlit run dashboard.py`).

They use compatible transaction fields, but remain separate workflows: the app
stores uploaded data in the current Streamlit session, while the processor
writes one headless dataset per person.

## Repository layout

| Path | Contents |
|------|----------|
| `dashboard.py`, `app.py`, `pages/` | The Streamlit app (Home / Process statements / Overview / Analytics / Budget) |
| `dashboard/` | The app's shared modules: import, export/validation, analytics, description cleaning, fuzzy search, budget PDF/Excel export, bank layout profiles |
| `ml/` | Headless dataset builder (`bank_statment_processor.py`) and ML experiments (`ml_models.py`) |
| `docs/` | `project-plan.md`, `report.md`, `CATEGORY_GUIDE.md` |
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
# Build clean, UNLABELED datasets (one per person) from .data/
uv run python -m ml.bank_statment_processor
```

Outputs:

* `savanna_transactions.csv` and `jillian_transactions.csv` – columns:
  `date, description, type, amount, main_category,
  sub_category, bank, account`.
  The description is the original bank text; the processor keeps a separate
  internal duplicate-review key but does not export it.
* `main_category` / `sub_category` start **blank on purpose**. Fill them in by
  hand (the app is the easiest way) or with the ML models once implemented.

## 4. Streamlit app (interactive labeling)

```bash
uv run streamlit run dashboard.py
```

Pages:

* **Home** – overview of the workflow.
* **Process statements** – upload bank CSVs, pick bank/column mapping, review
  rows, optionally assign category pairs, and download the CSV. Categories are
  never auto-filled; rows may remain uncategorized, or you can label them in
  the review panel. To continue in a later session, upload the prior website
  CSV download and choose **Load previous download and resume labeling**.
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

5. Cross-person experiment (train on Savanna's file, test on Jillian's):

```bash
uv run python -c "from ml.ml_models import cross_person_experiment; cross_person_experiment()"
```

6. Compare against the baselines (`baseline_most_common`) — see
   `CATEGORY_GUIDE.md` and `report.md` for the category vocabulary and results.

## 6. Using your own bank exports (data instructions)

* Put each person's exports under `.data/<person>/<bank>/` (for example,
  `.data/savanna/sofi/sofi_checking.csv`). Person folders are `savanna` and
  `jillian`.
* Known layouts (Goldenwest, SoFi, Capital One) are auto-detected by filename;
  the Streamlit app also has a column-mapping UI for other banks.
* **Privacy:** raw statements and the labeled dataset are private per our plan —
  do not commit `.data/` or labeled CSVs. To reproduce, run the pipeline on
  your own bank exports.

### Safe private-data workflow

1. Copy the bank exports into the correct private folders. Do not put them in
   `docs/`, `ml/`, `results/`, or the repository root:

   ```text
   .data\
     savanna\
       sofi\sofi_checking.csv
       goldenwest\goldenwest_credit.csv
     jillian\
       bank of america\boa_credit_card.csv
   ```

2. From the repository root, build the two standardized, unlabeled files:

   ```powershell
   uv run python -m ml.bank_statment_processor
   ```

   This creates `savanna_transactions.csv` and `jillian_transactions.csv`.
   Open those files in the labeling workflow, fill in `main_category` and
   `sub_category`, and save the labeled copies somewhere private. The
   processor's generated CSVs are ignored by Git.

3. Before running the experiment, confirm that both labeled files are local
   and that the required columns are present:

   ```powershell
   uv run python -c "import pandas as pd; from pathlib import Path; files=['jillian_transactions_labeled.csv','savanna_transactions.csv']; required={'date','description','type','amount','main_category','sub_category','bank','account'}; [print(f'{f}: {len(pd.read_csv(f))} rows') or (_ for _ in ()).throw(ValueError(f'{f} is missing {required-set(pd.read_csv(f).columns)}')) for f in files if Path(f).exists()]"
   ```

4. Run the first-result experiment. It writes only aggregate scores,
   category-level metrics, and confusion matrices to
   `results/first_result_nov05.md`; it does not write descriptions or amounts:

   **PowerShell:**

   ```powershell
   uv run python -m ml.first_result `
     --jillian jillian_transactions_labeled.csv `
     --savanna savanna_transactions.csv
   ```

   **Bash / Git Bash:**

   ```bash
   uv run python -m ml.first_result \
     --jillian jillian_transactions_labeled.csv \
     --savanna savanna_transactions.csv
   ```

5. Before committing documentation or code, verify that private files are
   ignored. The command should print nothing for the private CSV names:

   ```powershell
   git check-ignore -v ".data\savanna\sofi\sofi_checking.csv" `
     ".data\jillian\bank of america\boa_credit_card.csv" `
     "jillian_transactions_labeled.csv" `
     "savanna_transactions.csv"
   git status --short
   ```

   `git check-ignore` should print an ignore rule for each path. The private
   files should not appear in `git status --short`. Do not use `git add -f` on
   private files. If one appears in `git status`, stop and remove it from the
   staging area before committing.

## 7. Tests

```bash
uv run python -m unittest discover -s tests -v
```

Run after any change that touches importing, onboarding, or export logic.

## 8. Common issues

* `streamlit` not found → run `uv sync` first.
* App rejects a download → one or more rows have missing required transaction
  fields, an invalid date or amount, or only half of a category pair. Categories
  may be left blank, but Main Category and Sub Category must be filled together.
* New bank CSV won't parse → use the app's "Other / new bank" flow and map the
  columns; then note the layout below `dashboard/bank_profiles.json`.