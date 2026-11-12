# Transaction Categorization Dataset

STAT 486 Fall 2026 project – Savanna & Jillian

This project builds on the import approach from [Streamlit-Budget-Dashboard](https://github.com/Xapamma/Streamlit-Budget-Dashboard), but replaces its rule-based categorization.

**Route:** C – Build a dataset and use it in a reproducible ML pipeline  
**Project plan:** [docs/project-plan.md](docs/project-plan.md)

## Data and privacy

The data comes from our personal bank statement exports. Raw statements and labeled transactions are private and are not committed to this repository. To reproduce the pipeline, follow [SETUP.md](SETUP.md) with your own bank exports.

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
- Output, with per-category scores and confusion matrices: [results/first_result_nov05.md](results/first_result_nov05.md)
- Rerun from the project root (the labeled CSVs are private):
  ```bash
  uv run python -m ml.first_result --jillian jillian_transactions_labeled.csv --savanna savanna_transactions.csv
  ```

**What it suggests.** Within each person, KNN clearly beats the baseline on macro F1, so it learns the smaller categories. Most of its errors involve Transportation: 45 of Jillian's Food & Dining rows and 83 of Savanna's Savings & Investments rows were predicted as Transportation. Cross-person transfer is much weaker. No description appears in both people's data, and Jillian's data has no Gifts & Donations, Housing & Bills, or Savings & Investments rows, which make up 19% of Savanna's.

**Next.** Inspect the Transportation errors (features vs. labeling). Try distance-weighted KNN, tuning k, and word vs. character n-grams. Try logistic regression. Align our category sets. Add Jillian's checking/debit statements so both people cover the same kinds of transactions.

### 2026-11-12 – Working investigation

**What changed in the data.** We added Jillian's Bank of America checking/debit account for 2025 (statement dates 2024-12-30 to 2025-12-29), as planned on Nov 5. We then applied the same internal-transfer rule the processor uses for Savanna's data, dropping 88 rows (52 "Keep the Change" round-ups, 24 transfers to and from savings, and 12 credit card payments). Jillian's dataset now has 580 labeled rows (393 credit card + 187 checking), up from 393. Each row is still one bank transaction: date, bank description, signed amount, main category, sub-category, bank, and account. Labeling decisions: tuition eChecks are `Education / Tuition`. Venmo payments all share one description, so they get one label (`Food & Dining`, their most common use). 

Run order (raw and labeled files are private):

```bash
uv run python -m ml.bank_statment_processor          # 1. standardize bank exports in .data/<person>/<bank>/
# 2. label main/sub categories by hand using docs/CATEGORY_GUIDE.md
uv run python -m ml.quality_checks jillian --labeled jillian_transactions_labeled.csv --output results/data_quality_jillian_nov12.md
uv run python -m ml.first_result --jillian jillian_transactions_labeled.csv --savanna savanna_transactions.csv --output results/first_result_nov12.md
uv run python -m ml.confusion_analysis --jillian jillian_transactions_labeled.csv --savanna savanna_transactions.csv
```

Quality checks ([results/data_quality_jillian_nov12.md](results/data_quality_jillian_nov12.md)): no missing values, 1 possible duplicate (kept and flagged), no empty months, and the longest gap between transactions is 8 days. Food & Dining is now 53% of Jillian's rows (down from 62%). [`ml/quality_checks.py`](ml/quality_checks.py) now works with the processor's current column layout.

**A fairer evaluation split.** Within-person CV groups folds by description so the model is tested on descriptions it hasn't seen. But many bank descriptions differ only by an ID, a date, or a confirmation number (Venmo IDs, "Keep the Change ... FOR 02/03/25"). Those near-copies were landing in both train and test. [`ml/confusion_analysis.py`](ml/confusion_analysis.py) also groups by the description with digits and punctuation removed (`normalized`). This lowers Jillian's embedding-KNN accuracy from 0.672 to 0.586, so we report the normalized numbers from now on.

| Within-person, normalized grouping (5-fold mean) | Accuracy | Macro F1 |
|---|---|---|
| Jillian: most-common baseline | 0.534 | 0.080 |
| Jillian: TF-IDF KNN | 0.481 | 0.329 |
| Jillian: embedding KNN | **0.586** | **0.385** |
| Savanna: most-common baseline | 0.348 | 0.045 |
| Savanna: TF-IDF KNN | 0.527 | 0.330 |
| Savanna: embedding KNN | **0.596** | 0.330 |

Cross-person (embedding KNN, accuracy / macro F1): train Savanna → test Jillian 0.531 / 0.193 (baseline 0.534 / 0.070); train Jillian → test Savanna 0.307 / 0.105 (baseline 0.348 / 0.040). Full output: [results/first_result_nov12.md](results/first_result_nov12.md) and [results/confusion_analysis.md](results/confusion_analysis.md). The Nov 5 files ([results/first_result_nov05.md](results/first_result_nov05.md), [results/data_quality_jillian.md](results/data_quality_jillian.md)) are kept as submitted; their numbers differ because they use Jillian's 393 credit-card rows and exact-description grouping.

**TF-IDF vs. sentence embeddings.** Embeddings are the better default. They have higher accuracy for both of us and win 4 of Jillian's 10 categories (TF-IDF wins 1) and 7 of Savanna's 13 (TF-IDF wins 4). The clearest case is Education for Jillian (F1 0.59 vs 0.34): TF-IDF sends 72 Food & Dining rows to Education, spread across ordinary merchants (Venmo 14, WinCo 7, Chick-fil-A 6, Trader Joe's 5). Because no single merchant explains it, this is a general TF-IDF weakness, not one confusing merchant. For Savanna, macro F1 is tied at 0.330. TF-IDF still does better on Shopping, Health, Travel, and Gifts, and it makes far fewer Shopping ↔ Food & Dining errors (12 vs 60). So embeddings help most on large categories, not the small ones.

**What the biggest confusions show.** The largest errors for both of us come from one merchant used for two purposes, where only the amount tells them apart. Jillian has 47 small ($1–6) Chevron snack purchases next to 15 Chevron fuel purchases (54 Food → Transportation errors). Savanna has small snack purchases at Home Depot alongside home-improvement purchases, and snacks vs. fuel at Maverik (39 Transportation → Food & Dining errors, 29 Food & Dining → Shopping). Because the merchant is the same, normalized grouping puts all of these rows in one test fold, and the model never learns from a same-merchant example. The other big sources: Jillian's paychecks predicted as Transfer (14), her gym membership predicted as Education (12), and Savanna's generic "WEB AUTHORIZED PMT" description, which covers rent, utilities, a car payment, and transfers (24 errors). That last one is a data limitation, not a model problem.

**Plan changes since Nov 5.** (1) We added Jillian's checking data. (2) We dropped the plan to commit a made-up sample; the pipeline is reproduced by running it on your own bank exports ([SETUP.md](SETUP.md)). (3) The shared-label agreement check is now optional: Jillian prepared a 49-row sheet of her transactions with her labels for Savanna to label independently, weighted toward rows the models got wrong. (4) We now report the normalized grouping.

**Next question and what is uncertain.** Can the amount separate same-merchant cases (Chevron snacks vs. fuel, Home Depot snacks vs. supplies) without hurting other categories? We will tune the amount weight (currently ×25), try log-scaled amounts, and try logistic regression, which can learn merchant-by-amount combinations that KNN cannot. If small categories stay near F1 = 0, we will merge or drop them and focus on the main categories.
