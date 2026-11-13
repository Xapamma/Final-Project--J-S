# STAT 486 Project Plan

**Due:** Tuesday, October 27, 2026, at 11:59 PM Mountain Time  
**Team:** Person A: Savanna McLelland, Person B: Jillian Baker  
**Repository:** https\://github.com/Xapamma/Final-Project--J-S.git  
**Date updated:** October 27, 2026

## Project question and purpose

- **Question:** After cleaning and combining our personal bank statements into one dataset, which machine learning techniques most accurately categorize the transactions into main categories and subcategories? And can a model trained on one person’s bank statements correctly categorize the statements of another?
- **Why it interests us:** We built a Streamlit budgeting dashboard ([Streamlit-Budget-Dashboard](https://github.com/Xapamma/Streamlit-Budget-Dashboard)) that categorizes transactions with hand-written rules. We want to replace those rules with machine learning, and that starts with a clean, consistent dataset. Bank exports are messy and differ between banks, so building a reliable dataset is a real part of the work.
- **Expectation:** We expect the original transaction descriptions and amounts to provide useful signals for categorization, and we expect a model trained on one person to work only partly on the other because we shop at different places and use different banks. The results could also show that differences in how we each categorize transactions matter more than the model.

## Chosen route

- [x] **C. Build a dataset** and use it in a reproducible ML pipeline

**Why this route:** Our main contribution is the resulting categorized dataset that can be used to analyze personal spending. This includes the set of categories, a repeatable way to clean exports from different banks into one format, and correct categories for a set of both team members’ transactions so we can train and evaluate models. We will compare several standard ML methods, both supervised and unsupervised, to test whether the dataset is sufficient to categorize the transactions.

## Data source and access

- **Data:** CSV statement exports that each of us downloads from our own bank and credit-card accounts. Person A \= Savanna (training data); Person B \=Jillian (test data).
- **Relevant fields:** date, description, amount, and which bank and account type each transaction came from.
- **Access:** We download our own statements manually; no scraping or API is needed. The data is small enough to run on a laptop.
- **Use limits and ethics:** This is private financial data about the two of us. Raw statements and processed data **will not be committed to GitHub**. We will commit our code and a small made-up sample in the same format so others can run the pipeline. We will confirm this approach with the instructor.
- **Data quality concerns:** different column layouts and sign conventions across banks; duplicate rows from overlapping downloads; transfers between our own accounts that could look like spending; messy descriptions (store numbers, prefixes, locations); some categories having few examples; and the two of us possibly labeling similar purchases differently.

## Route-specific plan

### Route C: New dataset and ML pipeline

- **Dataset and question:** A labeled transaction dataset for two people across several banks, used to test how well a categorizer trained on one person works on the other as well as what ML method categorizes the transactions most accurately.
- **Sources:** Only our own statement exports. We may use our own data but will not redistribute it.
- **Observation and main fields:** one transaction per row, with person (A or B), bank, account type, date, original description, amount, and category label.
- **Building the dataset:**
  1. Load each bank's CSV into a common format, building on the dashboard's existing import code.
  2. Clean the data: standardize dates and signs, remove duplicates, flag transfers between our own accounts, and use the original descriptions for modeling.
  3. Write a short category guide so we label consistently, then each label our own transactions. We will both label a shared sample to check how consistently we label.
  4. Combine everything into one final dataset.
- **Documentation:** We will record which files were used (bank, account, date range, download date) and what each cleaning step changed.
- **Quality checks:** duplicates, missing values, gaps in dates, how many transactions each cleaning step affects, how many transactions fall in each category, and how often we agree on the shared sample.
- **Sharing:** The real data stays private. A made-up sample will let readers run the full pipeline, and anyone could rebuild a similar dataset from their own bank exports.
- **ML method:** All planned ML models will use pretrained sentence embeddings
  as their shared text representation. In the first run, KNN will use those
  embeddings plus amount, with character TF-IDF plus amount as an additional
  KNN-only comparison. Random Forests and ANN will use the embedding features.

## Initial reference and proposed contribution

- **Baseline:** always predicting the most common category. Models predict from the transaction description and amount (we dropped the merchant column).
- **Our contribution:** the category guide, the cleaning pipeline for multiple banks, the labeled dataset, and experiments showing what affects categorization accuracy.
- **Credit:** The import approach comes from Person A's existing dashboard repository (linked). We will use scikit-learn and pandas.

## Evaluation and evidence

- **Design:** First test within one person's data, making sure the same transaction doesn't appear in both training and test data. Then train on Person A's data and test on Person B's.
- **Comparisons:** the first run compares embedding KNN and TF-IDF KNN against
  the most-common baseline. Future Random Forest and ANN models will use the
  same embedding-plus-amount inputs as the embedding KNN.
- **Metric:** accuracy, plus checking how well each category does on its own
- **Evidence against our expectation:** cleaning that doesn't help, or results on Person B's data that are about as good as on Person A's.
- **Errors to look at:** which categories get confused (confusion matrix), merchants that only appear in Person B's data, and cases where the "error" is really a labeling difference between us.
- **Outputs:** a summary of the dataset, a results table, a confusion matrix, and a few anonymized examples.

## Risks and backup scope

- **Risks:** labeling takes longer than expected; some categories have too few examples; a bank only allows a short download history or only provides PDFs; privacy limits what we can show.
- **Backup scope:** use main categories only, use a shorter date range, or focus on one person's data and the effect of cleaning.
- **Pivots:** If evidence or feedback changes our direction, we will explain why in a dated README entry.

## Reproducibility and team contributions

- **Repository:** code, data instructions (no real data), results, and `report.md` will live in our new project repository, with `pyproject.toml`, `uv.lock`, and the Python version committed.
- **First runnable result (Nov. 5):** load the first labeled batch from each
  person, report basic quality checks, and compare sentence-embedding KNN
  with weighted amount against the additional character TF-IDF KNN
  comparison with weighted amount.
- **Team contributions:** Savanna leads training; Jillian leads testing. We will write the category guide together, label our own data, label a shared sample, and review each other's code.
- **Instructor feedback requested:** confirmation that using our own private financial data is fine with these precautions, and whether this dataset work is substantial enough for Route C.

## Sources

- Streamlit Budget Dashboard (Person A's prior project), [https\://github.com/Xapamma/Streamlit-Budget-Dashboard](https://github.com/Xapamma/Streamlit-Budget-Dashboard)
- Our own bank and credit-card statement exports (private; not shared)
- scikit-learn, [https\://scikit-learn.org](https://scikit-learn.org)
- pandas, [https\://pandas.pydata.org](https://pandas.pydata.org)
