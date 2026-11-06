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
| `ml/` | Headless dataset pipeline (`bank_statment_processor.py`) + ML experiments (`ml_models.py`, `first_result.py`) |
| `results/` | Generated outputs: data-quality checks and model results (category names and counts only, no transactions) |
| `docs/` | `project-plan.md`, `report.md`, `CATEGORY_GUIDE.md` |
| `tests/` | Unit tests |
| `.data/` | Private raw bank CSVs (gitignored) |

## Project log

### 2026-10-27 – Project plan

Route C: We are building a cleaned dataset of both teammates' bank transactions across multiple banks.

ML question: After cleaning and combining our personal bank statements into one dataset, which machine learning techniques most accurately categorize the transactions into main categories and subcategories? And can a model trained on one person’s bank statements correctly categorize the statements of another?

Raw data stays private; a made-up sample will support reproduction. Backup scope: main categories only, a shorter date range, or one person's data.

Full plan: [docs/project-plan.md](docs/project-plan.md)

### 2026-11-05 – Data and reference result

**Data.** Each of us exported our own transactions as CSVs from our banks' websites. Jillian's data is one Bank of America credit card (Dec 2024 to Dec 2025, 393 transactions). Savanna's comes from 4 banks and 7 accounts in 2025 (998 transactions). Both sets are fully hand-labeled using [docs/CATEGORY_GUIDE.md](docs/CATEGORY_GUIDE.md). [`ml/bank_statment_processor.py`](ml/bank_statment_processor.py) converts every bank's layout into one shared format. Raw statements (in `.data/`) and the labeled CSVs are gitignored and never committed. Categories are unbalanced (Food & Dining is 62% of Jillian's rows and 35% of Savanna's), so we report macro F1 alongside accuracy. Data checks ([`ml/quality_checks.py`](ml/quality_checks.py)): [Jillian](results/data_quality_jillian.md) and [Savanna](results/data_quality_savanna.md). Full checkpoint write-up: [docs/data_and_reference_result.md](docs/data_and_reference_result.md).

Rebuild the unlabeled dataset (standardizes bank formats, spending negative and income positive, categories left blank for labeling):

```bash
uv sync
uv run python -m ml.bank_statment_processor
```

**First result.** We predict `main_category` from the original transaction
description and amount only (we dropped the merchant column). The most-common
category is the simple baseline. All planned ML models use pretrained
`all-MiniLM-L6-v2` sentence embeddings plus amount; the first run also
includes a TF-IDF-plus-amount KNN comparison. Amounts are standardized using
training data and multiplied by 25. KNN uses cosine distance with `k = 5`.
Within-person scores use 5-fold GroupKFold grouped by description, so test
descriptions never appear in training. Cross-person scores train on one
person and test on the other. The checked-in result table below contains the current comparison.

| Setting | Most-common baseline | TF-IDF KNN | Embedding KNN |
|---|---|---|---|
| Jillian within-person | 0.624 | 0.483 | 0.603 |
| Savanna within-person | 0.348 | 0.505 | 0.559 |
| Train Savanna → test Jillian | 0.623 | 0.384 | 0.496 |
| Train Jillian → test Savanna | 0.348 | 0.185 | 0.274 |

- Code: [`ml/first_result.py`](ml/first_result.py) (runner) and [`ml/ml_models.py`](ml/ml_models.py) (baseline, KNN, cross-validation, cross-person functions). Tests: [`tests/test_first_result.py`](tests/test_first_result.py)
- Output, with per-category scores and confusion matrices: [results/first_result.md](results/first_result.md)
- Rerun from the project root (the labeled CSVs are private):
  ```bash
  uv run python -m ml.first_result --jillian jillian_transactions_labeled.csv --savanna savanna_transactions.csv
  ```

**What it suggests.** Within each person, KNN clearly beats the baseline on macro F1, so it learns the smaller categories. Most of its errors involve Transportation: 45 of Jillian's Food & Dining rows and 83 of Savanna's Savings & Investments rows were predicted as Transportation. Cross-person transfer is much weaker. No description appears in both people's data, and Jillian's data has no Gifts & Donations, Housing & Bills, or Savings & Investments rows, which make up 19% of Savanna's.

**Next.** Inspect the Transportation errors (features vs. labeling). Try distance-weighted KNN, tuning k, and word vs. character n-grams. Try logistic regression. Align our category sets. Add Jillian's checking/debit statements so both people cover the same kinds of transactions.
