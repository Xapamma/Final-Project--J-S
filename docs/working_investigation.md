# Working Investigation

**Checkpoint:** November 12, 2026, by 11:59 PM Mountain Time  
**Team:** Savanna McLelland & Jillian Baker
**Route:** C 
**Date:** November 11th, 2026

Use the prompts for your chosen route to describe what is working now and what you learned from a first check. Add a new, dated entry for the working investigation to the root `README.md` of your team's GitHub repository and link the relevant code and output there. Keep earlier entries, commit the update, and submit the repository URL with that commit in Canvas. This file is optional supporting material; it does not replace the README entry. Preliminary or negative results are welcome when you explain what they show.

The graded entry is the "2026-11-12 – Working investigation" section of the [README](../README.md). The answers below follow the prompts and use the same numbers.

## Working code and evidence

- **What part of your investigation runs now? Link the code, configuration, and an output it produced.**

  The full pipeline runs end to end, from bank exports to the model comparison:
  - [`ml/bank_statment_processor.py`](../ml/bank_statment_processor.py) standardizes the bank exports.
  - [`ml/quality_checks.py`](../ml/quality_checks.py) runs the data checks.
  - [`ml/first_result.py`](../ml/first_result.py) and [`ml/ml_models.py`](../ml/ml_models.py) run the baseline and the two KNN models.
  - [`ml/confusion_analysis.py`](../ml/confusion_analysis.py) finds the biggest confusion pairs.

  Outputs: [`results/first_result_nov12.md`](../results/first_result_nov12.md), [`results/confusion_analysis.md`](../results/confusion_analysis.md), and [`results/data_quality_jillian_nov12.md`](../results/data_quality_jillian_nov12.md).

- **What inputs and steps does the run use? Give the command to reproduce the first check if available, and note any data access or setup needed.**

  Each script takes our hand-labeled transaction CSVs as input. Raw statements and labeled files are private and gitignored, so to reproduce, follow [SETUP.md](../SETUP.md) with your own bank exports and run:

  ```bash
  uv run python -m ml.bank_statment_processor
  # label main/sub categories using docs/CATEGORY_GUIDE.md
  uv run python -m ml.quality_checks jillian --labeled jillian_transactions_labeled.csv --output results/data_quality_jillian_nov12.md
  uv run python -m ml.first_result --jillian jillian_transactions_labeled.csv --savanna savanna_transactions.csv --output results/first_result_nov12.md
  uv run python -m ml.confusion_analysis --jillian jillian_transactions_labeled.csv --savanna savanna_transactions.csv
  ```

- **What comparison or diagnostic did you run, using which evaluation split, metric, or quality measure? What does the result show so far?**

  We compared three models for predicting `main_category`:
  - a most-common-category baseline,
  - TF-IDF KNN (character n-grams),
  - sentence-embedding KNN (all-MiniLM-L6-v2).

  Both KNN models also use the standardized amount. Metrics are accuracy and macro F1 from 5-fold GroupKFold within each person, plus training on one person and testing on the other.

  We found that rows differing only by an ID, date, or confirmation number were landing in both train and test. We now group folds by the description with digits and punctuation removed. This lowered Jillian's embedding-KNN accuracy from 0.672 to 0.586.

  | Within-person, normalized grouping | Accuracy | Macro F1 |
  |---|---|---|
  | Jillian: baseline / TF-IDF / embeddings | 0.534 / 0.481 / 0.586 | 0.080 / 0.329 / 0.385 |
  | Savanna: baseline / TF-IDF / embeddings | 0.348 / 0.527 / 0.596 | 0.045 / 0.330 / 0.330 |

  Embeddings are the better default, but the margin over the baseline is small for Jillian. TF-IDF still makes far fewer of Savanna's Shopping ↔ Food & Dining errors (12 vs 60). The biggest errors for both of us come from one merchant used for two purposes, where only the amount tells them apart: gas-station snacks vs. fuel, and Home Depot snacks vs. supplies.

### Route C: New dataset and ML pipeline

- **Where are the collection, cleaning, and integration scripts that produce the cleaned dataset? Give the run order and record the source dates or versions and any access limits.**

  The run order is the command block above. Each of us exported our own statements.

  | Person | Accounts | Dates |
  |---|---|---|
  | Jillian | Bank of America credit card | 2024-12-05 to 2025-12-03 |
  | Jillian | Bank of America checking/debit | 2024-12-30 to 2025-12-29 |
  | Savanna | 4 banks, 7 accounts | 2025 |

  Jillian's checking statements were checked against the bank's printed section totals, and all 36 sections matched to the cent. Access limit: this is personal financial data, so nothing raw or labeled is committed.

- **What does one cleaned observation represent? Link a permissible sample or summary and report relevant quality checks, such as coverage, duplicates, missingness, or label consistency.**

  One row is one bank transaction with these fields: date, bank description, signed amount (spending negative), main category, sub-category, bank, and account. Summaries: [Jillian](../results/data_quality_jillian_nov12.md) and [Savanna](../results/data_quality_savanna.md).

  Jillian's quality checks:
  - 668 raw rows. 88 internal transfers dropped (round-ups, savings transfers, card payments), leaving 580.
  - 0 noise rows remaining.
  - No missing values.
  - 1 possible duplicate, kept and flagged.
  - No empty months. The longest gap between transactions is 8 days.
  - Food & Dining is 53% of the rows.

  Label consistency: we fixed the "Tutition" typo and wrote down labeling decisions in the README. An optional shared-labeling agreement check was offered.

- **What initial ML result did you obtain using the cleaned output of this pipeline? Link its code and output. What do the quality checks and ML result suggest you should investigate next?**

  See the table above, [`results/first_result_nov12.md`](../results/first_result_nov12.md), and [`results/confusion_analysis.md`](../results/confusion_analysis.md).

  What to investigate next:
  - **The amount feature.** The same-merchant confusions can only be fixed using the amount.
  - **Small categories.** Some have fewer than 10 rows and F1 near 0. They need more data or merging.
  - **Generic descriptions.** Savanna's "WEB AUTHORIZED PMT" covers several categories (24 errors), and the description alone can't separate them.

## Next question

- **If your plan changed since November 5, what evidence or feedback led to the change?**

  1. We added Jillian's checking data as planned, so both people cover similar kinds of transactions.
  2. We dropped the made-up sample. The pipeline is reproduced with your own bank exports instead.
  3. The shared-label agreement check is now optional.
  4. We report the normalized grouping, because the error analysis showed near-duplicate descriptions in both train and test.

- **What is uncertain or blocked now, and how will you adjust the scope if needed?**

  Next question: can the amount separate same-merchant cases without hurting other categories? We will tune the amount weight (currently ×25), try log-scaled amounts, and try logistic regression.

  Still uncertain:
  - Our category sets differ. Jillian has no Housing & Bills, Savings & Investments, or Gifts & Donations rows.
  - Several categories are very small.
  - All Venmo payments share one label.

  If small categories stay near F1 = 0, we will merge or drop them and focus on the main categories.
