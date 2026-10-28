# Project Report — Transaction Categorization with Machine Learning

**STAT 486 Fall 2026** · Savanna McLelland & Jillian Baker
**Route:** C – Build a dataset and use it in a reproducible ML pipeline
**Date updated:** October 27, 2026

> This is a living report. Results sections are placeholders until our first
> runnable ML results (target: Nov 5, see § 8). All numbers we add must come
> from actual runs, not estimates.

## 1. Project question

After cleaning and combining our personal bank statements into one dataset,
**which machine learning techniques most accurately categorize the
transactions into main categories and subcategories?** And **can a model
trained on one person's bank statements correctly categorize the statements of
another?**

## 2. Motivation

Our earlier Streamlit budgeting dashboard
([Streamlit-Budget-Dashboard](https://github.com/Xapamma/Streamlit-Budget-Dashboard))
categorized transactions with hand-written rules and a merchant cache. We
wanted to replace that static system with machine learning, and build a clean,
labeled dataset — the hard part — along the way.

**Expectation:** cleaning descriptions should help categorization, a model
trained on one person will only partly work on the other (different merchants,
banks), and label-style differences between us may matter as much as the model.

## 3. Methods

### 3.1 Dataset

* Sources: our personal CSV exports from Goldenwest Credit Union, SoFi, and
  Capital One (two people; raw data stays private).
* Building: `ml/bank_statment_processor.py` loads each bank's layout into one
  common schema (`date, description, raw_description, type, amount,
  main_category, sub_category, bank, account`).
* Cleaning: descriptions are normalized (lowercase, digits/symbols and bank
  noise stripped); internal transfers/roundups and declined rows are excluded
  as noise (recoverable in the app's review panel); duplicates and payment-like
  rows are flagged for review.
* **Categorization was intentionally removed**: `main_category` /
  `sub_category` start blank and are assigned by us (the Streamlit app is the
  labeling tool) using the vocabulary in `CATEGORY_GUIDE.md`. No rules, caches,
  or default labels are used.

> Status: 572 transactions combined locally on 2026-10-27, with categories
> blank pending labeling. **No real data is committed to the repository.**

### 3.2 Labeling

* We label our own transactions in the Streamlit app (`streamlit run
  dashboard.py`), which enforces a main::sub pair from the category guide.
* We each label a shared sample to measure agreement (see
  `CATEGORY_GUIDE.md` § 4).

### 3.3 ML methods (to be implemented in `ml/ml_models.py`)

* **Features:** TF-IDF text embeddings of `description` (we will also compare
  raw vs. cleaned text and adding amount/bank/account features).
* **Models:** k-nearest neighbors (KNN), Random Forest, and a simple neural
  network (ANN/MLP) via scikit-learn; each produces a `main_category` and a
  `sub_category` prediction.
* **Baselines to beat:** (1) always predicting the most common category,
  (2) per-description lookup of the most frequent category in training.

### 3.4 Evaluation design

1. Within-person: train/test split on one person's data, ensuring the same
   transaction never appears in both sides.
2. Cross-person: train on Person A's labels, test on Person B's, and vice
   versa (`cross_person_experiment()`).
3. Metric: overall accuracy plus per-category accuracy, with a confusion
   matrix; we report the effect of cleaning (raw vs. cleaned descriptions).

## 4. Results

### 4.1 Dataset summary (TBD after labeling)

| Item | Value |
|---|---|
| Transactions | _572 (unlabeled) / TBD_ |
| Banks / accounts | _TBD_ |
| Labels | _TBD_ |

### 4.2 Quality checks (TBD)

Duplicates removed, missing values, rows excluded per cleaning step, labels
per category, and shared-sample agreement will be summarized here.

### 4.3 Model accuracy (TBD after Nov 5 results)

| Method | Main accuracy | Sub accuracy | Beats baseline? |
|---|---|---|---|
| Baseline: most common | _TBD_ | _TBD_ | — |
| Baseline: per-description lookup | _TBD_ | _TBD_ | — |
| KNN | _TBD_ | _TBD_ | _TBD_ |
| Random Forest | _TBD_ | _TBD_ | _TBD_ |
| ANN / MLP | _TBD_ | _TBD_ | _TBD_ |

### 4.4 Cross-person (TBD)

| Trained on | Tested on | Main accuracy | Sub accuracy |
|---|---|---|---|
| A | B | _TBD_ | _TBD_ |
| B | A | _TBD_ | _TBD_ |

### 4.5 Errors & confusion (TBD)

Most-confused category pairs and any "error" that is really a labeling
difference between us (linked to `CATEGORY_GUIDE.md` § 4).

## 5. Discussion & limitations

_TBD after results._ Likely themes to cover:

* Does cleaning the descriptions actually help (raw vs. cleaned)?
* Do small categories hurt accuracy, and is main-category-only a better
  backup scope?
* How much of the cross-person gap is the model vs. how we label?
* Which merchants look different between us (never seen in training)?

## 6. Privacy & reproduction

* Raw statements and the labeled dataset stay private and are **not
  committed** to GitHub. Only code, docs, and a small made-up sample are
  committed so the pipeline can be reproduced.
* Reproduce with the made-up sample: follow `SETUP.md` (install → run the
  pipeline → train → evaluate). Anyone can substitute their own bank exports.
* All processing is local (see `AI_USAGE.md` for what the AI assistant
  touched during development).

## 7. AI assistance summary

We used an AI coding assistant (Claude/Cline) as a pair-programming assistant
during development. **What AI did:** removed the old rule-based categorizer;
created the `ml/ml_models.py` scaffolding/instructions (not the model code);
ported and de-categorized the Streamlit dashboard; added dependencies; wrote
the test suite; helped write this documentation. **What AI did not do:** design
the experiments, train models, interpret results, or make project decisions.

Full dated log and future-use policy: [`AI_USAGE.md`](AI_USAGE.md).

## 8. Project log

### 2026-10-27 – Project plan & first pipeline
* Wrote `project-plan.md`; created this repository.
* Removed the rule-based categorization system; categories now start blank for
  manual labeling (Streamlit app or CSV).
* Ported the Streamlit dashboard (Home / Process / Overview / Analytics /
  Budget) and de-categorized it; created `ml/ml_models.py` with implementation
  instructions; 31 tests passing.
* Added docs: `SETUP.md`, `CATEGORY_GUIDE.md`, `AI_USAGE.md`, this report.

### Nov 5 – First runnable ML result (target)
* Load and clean the first labeled batch; report basic quality checks.
* Add text features and try KNN; fill in § 4 tables with real numbers.

## 9. Next steps (checklist)

- [ ] Label transactions (split between us) + fill `CATEGORY_GUIDE.md` § 4.
- [ ] `uv add scikit-learn` and implement `ml/ml_models.py` TODOs (document any
      AI help in `AI_USAGE.md`).
- [ ] Run KNN (target Nov 5) using the instructions in `ml/ml_models.py`
      docstrings; then Random Forest and ANN.
- [ ] Run `cross_person_experiment()` and compare with baselines.
- [ ] Fill in § 4 tables and the confusion matrix from real runs.
- [ ] Confirm privacy and the AI policy with the instructor (per plan).