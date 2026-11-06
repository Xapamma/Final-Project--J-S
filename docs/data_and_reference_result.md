# Data and Reference Result

**Checkpoint:** November 5, 2026, by 11:59 PM Mountain Time

**Team:** Savanna & Jillian

**Date:** 11/2/2026

Use any prompts that help you describe your progress. Add a dated entry to the root README.md of your team's repository and link the relevant code and output there; this file is optional supporting material.

## Data

### Where did the data come from, and how did you obtain or prepare them?

Each of us exported our own transaction history as CSVs from our banks' websites: Jillian's is one Bank of America credit card (Dec 2024 to Dec 2025), and Savanna's comes from Savanna’s credit and debit cards from Goldenwest, Sofi, Capital one. The raw files stay in a gitignored `.data/` folder and are never committed. `ml/bank_statment_processor.py` converts every bank's layout into one shared format and makes signs consistent (spending negative, money in positive). We then label each transaction's main and sub category by hand using `docs/CATEGORY_GUIDE.md` (rerun: `uv sync`, then `uv run python -m ml.bank_statment_processor`).

### What data are available to your team now?

We currently have Jillian's 393 Bank of America credit-card transactions all labeled, and Savanna's 999 transactions from 4 banks of them labeled so far.

### What remains to be collected or cleaned?

We need to build the made-up sample and decide how to combine both datasets privately for the cross-person test.

### Route C: What does your first collected and cleaned sample contain?

Jillian's sample has 393 transactions (377 debits, 16 credits) from one credit card over 13 months, with 163 distinct merchants. Savanna's has 998 transactions (863 debits, 135 credits) across 4 banks and 7 accounts in 2025, with 423 distinct merchants, all labeled. Neither has missing values or months without transactions. Categories are unbalanced in both (Food & Dining is 62% of Jillian's and 35% of Savanna's), so we report macro-F1 alongside accuracy; full checks are in `results/data_quality_<name>.md`

## Evaluation plan

### What comparison, split, metric, or diagnostic will help answer your question? Why?

We will compare several machine learning models, including KNN, Random Forest, and a neural network, against a baseline that always predicts the most common category in the training data. We will use cross-validation within Savanna’s training data to compare models and select settings, reporting average scores and variation across folds. We will then evaluate the selected model on held-out transactions from Savanna and on Jillian’s transactions to assess performance within and across people. We will also compare raw and cleaned descriptions using the same splits to measure whether cleaning improves categorization. We will report accuracy and macro F1 separately for main categories and subcategories. Accuracy measures overall correctness, while macro F1 gives each category equal weight so that common categories do not dominate the results. Confusion matrices and separate results for familiar versus unfamiliar merchants will help explain where the models struggle.

### How will you avoid data leakage or another misleading comparison?

We will carefully compile our datasets so that the same transaction does not appear in both training and evaluation data. We will use text embeddings to convert transaction descriptions into numerical vectors that represent their meaning. If we train or fine-tune the embedding model, we will do so only on training data, including within each cross-validation fold. Other learned preprocessing will also use only training data, and category labels will never be included as predictors. We will select model settings through cross-validation on Savanna’s training data before evaluating on the held-out datasets. All models and cleaning comparisons will use the same evaluation splits to ensure a fair comparison. We will also follow a shared category guide and check labeling agreement so that differences in our preferences are not mistaken for model errors. Finally, we will distinguish performance on familiar and unfamiliar merchants and acknowledge that results from two people do not establish performance for all users.

## First runnable result

### What baseline or other reference did you run? What did it produce?

For our baseline, we used a model that always predicts the most common category in the training data. For the training data (Savanna’s dataset) that was ….

### Where are the code and output? Link them, and include the command needed to rerun the result if available.

### What did this first result suggest, and what will you try next?
