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

**Expectation:** models should generalize better when descriptions have useful
shared text patterns, a model trained on one person will only partly work on
the other (different descriptions and banks), and label-style differences
between us may matter as much as the model.

## 3. Methods

### 3.1 Dataset

* Sources: our personal CSV exports from Goldenwest Credit Union, SoFi, and
  Capital One (two people; raw data stays private).
* Building: `ml/bank_statment_processor.py` loads each bank's layout into one
  common schema (`date, description, type, amount,
  main_category, sub_category, bank, account`).
* Cleaning: the original bank description is preserved; declined rows and
  duplicates are queued for review, while
  internal transfers, roundups, and payment-like rows remain editable for
  manual labeling.
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

### 3.3 ML methods

* **Baseline:** always predicting the most common category.
* **Shared text representation:** all planned ML models use pretrained
  `all-MiniLM-L6-v2` sentence embeddings plus the standardized amount
  weighted by 25. The embedding model is loaded as-is and is not fine-tuned
  on our transactions.
* **First-run KNN:** embedding KNN uses the shared embedding features, while
  TF-IDF KNN is an additional comparison using character-level `char_wb`
  3–5-grams plus the same weighted amount. Both use cosine distance and five
  neighbors.
* The most-common category is the baseline. Random Forest and ANN/MLP are
  planned models that will use the shared embedding features; they are not
  completed first-result models.

### 3.4 Evaluation design

1. Within-person: train/test split on one person's data, ensuring the same
   transaction never appears in both sides.
2. Cross-person: train on Person A's labels, test on Person B's, and vice
   versa (`cross_person_experiment()`).
3. Metric: overall accuracy plus macro-F1, per-category scores, and confusion
   matrices; the first run compares the most-common baseline, TF-IDF KNN, and
   embedding KNN on the same splits.

## 4. Results

### 4.1 Dataset summary

| Item | Value |
|---|---|
| Transactions | 1,391 labeled transactions: 393 Jillian, 998 Savanna |
| Banks / accounts | 5 banks / 8 accounts: Jillian (1 / 1), Savanna (4 / 7) |
| Labels | 1,391 / 1,391 rows labeled with main and subcategory |

### 4.2 Quality checks

Both datasets have complete values for all eight required columns and all
transactions are labeled. Jillian has no rows removed during processing and
one possible duplicate; Savanna has 77 possible duplicates flagged for review.
Neither dataset has months without transactions. Jillian's data covers one
Bank of America credit-card account, while Savanna's data covers seven
accounts across four banks. Savanna's raw processor cleaning counts were not
available for this checkpoint, and shared-sample labeling agreement was not
measured.

### 4.3 Model accuracy

The first runnable result predicts `main_category`; subcategory scores were
not produced by this run. Values below are within-person mean accuracy across
five grouped folds. The detailed per-category reports and confusion matrices
are in [`results/first_result.md`](../results/first_result.md).

| Method | Jillian accuracy | Savanna accuracy | Beats most-common baseline? |
|---|---|---|---|
| Baseline: most common | 0.624 | 0.348 | — |
| TF-IDF KNN | 0.483 | 0.505 | Jillian: No; Savanna: Yes |
| Embedding KNN | 0.603 | 0.559 | Jillian: No; Savanna: Yes |

### 4.4 Cross-person

The first runnable result evaluates `main_category`; subcategory accuracy was
not produced. The values below come from the cross-person experiment in
[`results/first_result.md`](../results/first_result.md).

| Trained on | Tested on | Most-common accuracy | TF-IDF KNN accuracy | Embedding KNN accuracy | Embedding macro-F1 |
|---|---|---:|---:|---:|---:|
| Savanna | Jillian | 0.623 | 0.384 | 0.496 | 0.135 |
| Jillian | Savanna | 0.348 | 0.185 | 0.274 | 0.089 |

### 4.5 Errors & confusion

Embedding KNN improved over TF-IDF KNN for Savanna (0.559 vs. 0.505) and
cross-person transfer from Savanna to Jillian (0.496 vs. 0.384), but not for
Jillian within-person (0.603 vs. 0.483) or transfer from Jillian to Savanna
(0.274 vs. 0.185). Macro-F1 shows the same general pattern while remaining
well below accuracy because the category distributions are highly imbalanced.
The detailed confusion matrices and per-category scores are in
[`results/first_result.md`](../results/first_result.md).

## 5. Discussion & limitations

Embedding KNN outperformed TF-IDF KNN for Savanna within-person
classification and for transfer from Savanna to Jillian, but not for Jillian
within-person classification or transfer from Jillian to Savanna. The
most-common baseline still had higher accuracy in both within-person Jillian
and cross-person tests because the category distributions are highly
imbalanced. Macro-F1 is therefore important: it shows that KNN improves
recognition of smaller categories even when overall accuracy is lower.

The cross-person results are substantially weaker than the within-person
results, and no transaction descriptions were shared between the two
datasets. This suggests that differences in merchants, account types, and
label distributions limit transfer. The experiment evaluates only
`main_category`; subcategory performance, shared-sample agreement, and the
planned Random Forest and ANN/MLP comparisons remain outside this first
result. The detailed category errors and confusion matrices are documented
in [`results/first_result.md`](../results/first_result.md).

## 6. Privacy & reproduction

* Raw statements and the labeled dataset stay private and are **not
  committed** to GitHub. Only code, docs, and a small made-up sample are
  committed so the pipeline can be reproduced.
* Reproduce with the made-up sample: follow `SETUP.md` (install → run the
  pipeline → train → evaluate). Anyone can substitute their own bank exports.
## 7. Project log

### 2026-10-27 – Project plan & first pipeline
* Wrote `project-plan.md`; created this repository.
* Removed the rule-based categorization system; categories now start blank for
  manual labeling (Streamlit app or CSV).
* Ported the Streamlit dashboard (Home / Process / Overview / Analytics /
  Budget) and de-categorized it; created `ml/ml_models.py` with implementation
  instructions; 31 tests passing.
* Added docs: `SETUP.md`, `CATEGORY_GUIDE.md`, and this report.

### Nov 5 – First runnable ML result
* Load the first labeled batches and report basic quality checks.
* Compare the most-common baseline, embedding-plus-amount KNN, and the
  additional TF-IDF-plus-amount KNN comparison.

## 8. Next steps (checklist)

- [ ] Label transactions (split between us) + fill `CATEGORY_GUIDE.md` § 4.
- [x] Add character TF-IDF KNN and embedding-plus-amount KNN comparisons.
- [ ] Implement and evaluate Random Forest and ANN/MLP comparisons.
- [ ] Run `cross_person_experiment()` and compare with the baseline.
- [ ] Fill in § 4 tables and the confusion matrix from real runs.
- [ ] Confirm privacy and reproducibility expectations with the instructor.