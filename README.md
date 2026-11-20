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
| `dashboard/` | Shared app modules: import, export/validation, analytics, search, budget export, bank profiles |
| `ml/` | Headless dataset pipeline (`bank_statment_processor.py`), data checks (`quality_checks.py`), and ML experiments (`ml_models.py`, `first_result.py`, `model_comparison.py`, `feature_ablation.py`, `make_figures.py`) |
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
The model always receives the original, uncleaned description text. For
within-person scores, 5-fold GroupKFold assigns all rows with the same exact
description to one fold, so an identical description cannot appear in both
training and test data. This prevents the model from being tested on a
description it memorized, without changing the text used as a feature.
Cross-person scores train on one person and test on the other. The checked-in
result table below contains the current comparison.

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
```

Quality checks ([results/data_quality_jillian_nov12.md](results/data_quality_jillian_nov12.md)): no missing values, 1 possible duplicate (kept and flagged), no empty months, and the longest gap between transactions is 8 days. Food & Dining is now 53% of Jillian's rows (down from 62%). [`ml/quality_checks.py`](ml/quality_checks.py) now works with the processor's current column layout.

**Evaluation split.** The model uses the original, uncleaned transaction
descriptions. Within-person CV uses exact-description grouping: all rows with
identical description text are assigned to the same fold, so the exact same
description cannot occur in both training and test data. This controls for
memorization while leaving the model input unchanged. The same grouping is
used by both result scripts.

| Within-person, exact-description grouping (5-fold mean) | Accuracy | Macro F1 |
|---|---|---|
| Jillian: most-common baseline | 0.534 | 0.080 |
| Jillian: TF-IDF KNN | 0.481 | 0.329 |
| Jillian: embedding KNN | **0.586** | **0.385** |
| Savanna: most-common baseline | 0.348 | 0.045 |
| Savanna: TF-IDF KNN | 0.527 | 0.330 |
| Savanna: embedding KNN | **0.596** | 0.330 |

Cross-person (embedding KNN, accuracy / macro F1): train Savanna → test Jillian 0.531 / 0.193 (baseline 0.534 / 0.070); train Jillian → test Savanna 0.307 / 0.105 (baseline 0.348 / 0.040). Full output: [results/first_result_nov12.md](results/first_result_nov12.md). The Nov 5 files ([results/first_result_nov05.md](results/first_result_nov05.md), [results/data_quality_jillian.md](results/data_quality_jillian.md)) are kept as submitted; their numbers differ because they use Jillian's 393 credit-card rows and exact-description grouping.

**TF-IDF vs. sentence embeddings.** Embeddings are the better default. They have higher accuracy for both of us and win 4 of Jillian's 10 categories (TF-IDF wins 1) and 7 of Savanna's 13 (TF-IDF wins 4). The clearest case is Education for Jillian (F1 0.59 vs 0.34): TF-IDF sends 72 Food & Dining rows to Education, spread across ordinary merchants (Venmo 14, WinCo 7, Chick-fil-A 6, Trader Joe's 5). Because no single merchant explains it, this is a general TF-IDF weakness, not one confusing merchant. For Savanna, macro F1 is tied at 0.330. TF-IDF still does better on Shopping, Health, Travel, and Gifts, and it makes far fewer Shopping ↔ Food & Dining errors (12 vs 60). So embeddings help most on large categories, not the small ones.

**What the biggest confusions show.** The largest errors for both of us come from one merchant used for two purposes, where only the amount tells them apart. Jillian has 47 small ($1–6) Chevron snack purchases next to 15 Chevron fuel purchases (54 Food → Transportation errors). Savanna has small snack purchases at Home Depot alongside home-improvement purchases, and snacks vs. fuel at Maverik (39 Transportation → Food & Dining errors, 29 Food & Dining → Shopping). The other big sources: Jillian's paychecks predicted as Transfer (14), her gym membership predicted as Education (12), and Savanna's generic "WEB AUTHORIZED PMT" description, which covers rent, utilities, a car payment, and transfers (24 errors). That last one is a data limitation, not a model problem.

**Plan changes since Nov 5.** (1) We added Jillian's checking data. (2) We dropped the plan to commit a made-up sample; the pipeline is reproduced by running it on your own bank exports ([SETUP.md](SETUP.md)). (3) We reviewed each other's transaction labels to check that we were applying the category guide consistently. (4) We use exact-description grouping consistently in the evaluation scripts.

**Next questions and what is uncertain.** Can transaction amount help distinguish purchases with similar descriptions, such as snacks and fuel from the same merchant, without hurting performance in other categories? We also need to determine how to improve predictions for rare categories while keeping them in the dataset. Low scores may reflect too few examples, inconsistent labels, or transaction descriptions that do not provide enough information. We will investigate approaches such as collecting more examples and oversampling rare categories in training, then examine category-level results to see whether they help. Cross-person performance also remains uncertain because our datasets cover different accounts and categories.

### 2026-11-19 – Evidence and interpretation

**Data quality.** Both datasets exclude internal transfers between a person's own accounts. The [Jillian quality report](results/data_quality_jillian_nov12.md) checks the exact 580 rows the models use (393 credit-card + 187 checking, after removing 88 internal transfers): no missing values, one possible duplicate, no empty months; Food & Dining is 53.4% of rows. The [Savanna report](results/data_quality_savanna.md) covers 998 labeled rows, with 77 possible duplicates (possibly real repeat purchases), no missing values in the reported fields, and no empty months. Savanna's report could not reconstruct cleaning-step counts from the available Streamlit export and raw files. Credit-card payments remain labeled `Transfer` in both (27 Jillian rows, 18 Savanna rows).

**What the models show.** We compared TF-IDF KNN, embedding KNN, random forest, and a neural network against a most-common-category baseline, using nested cross-validation with five description-grouped outer folds within each person ([tuned results](results/tables/tuned_comparison_nov19_scores.csv), [figure](results/media/macro_f1_comparison.svg)). For Savanna, the neural network had the highest accuracy and macro F1 (0.657 / 0.542 vs. 0.348 / 0.040 for the baseline). For Jillian, TF-IDF KNN did (0.745 / 0.623 vs. 0.534 / 0.070). On the same 580 Jillian rows, the untuned Nov 12 embedding KNN scored 0.586 / 0.385, so tuning and rare-category oversampling improved results substantially.

**Cross-person and features.** A model trained on one person does not reliably categorize the other's transactions ([cross-person table](results/tables/tuned_comparison_nov19_cross_person.csv)). Training on Savanna and testing on Jillian, no tuned model beat the baseline's accuracy (best 0.528 vs. 0.534), though TF-IDF KNN had higher macro F1 (0.286 vs. 0.070). Training on Jillian and testing on Savanna, embedding KNN reached 0.450 / 0.179 vs. 0.348 / 0.040. In the fixed-setting [feature comparison](results/tables/feature_ablation_scores.csv) ([figure](results/media/feature_amount_effect.svg)), adding amount improved within-person macro F1 in 7 of 8 person/model comparisons but only 5 of 8 cross-person comparisons. See the [final results and figures](results/final_results_nov19.md), [comparison code](ml/model_comparison.py), [ablation code](ml/feature_ablation.py), and [figure code](ml/make_figures.py).

**Informative error.** Training on Savanna and testing on Jillian, no tuned model beat the most-common baseline's accuracy (best 0.528 vs. 0.534), even though macro F1 was much higher (0.286 vs. 0.070). The models recognize some of Jillian's less common categories but miss enough of her dominant Food & Dining category (53% of her rows vs. 35% of Savanna's) to fall below a constant guess. So we conclude that a model trained on one person does not reliably transfer to another, and the result depends on whose data it was trained on.

**What changed and next step.** There is no single best model: a neural network led for Savanna and TF-IDF KNN for Jillian. Amount helped within each person (7 of 8 comparisons) but transferred inconsistently across people (5 of 8). Tuning and oversampling improved Jillian's scores substantially compared with Nov 12 (0.586 / 0.385 → 0.745 / 0.623). Next, we will train on the combined dataset, check whether amount separates same-merchant purchases like snacks vs. fuel, and evaluate subcategories. More detail: [evidence and interpretation](docs/evidence-and-interpretation.md).
