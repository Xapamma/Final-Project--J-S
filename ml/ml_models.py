"""
ml_models.py
============
Our machine-learning categorization systems live here.

WHY THIS FILE EXISTS
--------------------
We removed the old rule-based categorizer (description_map,
sub_category_map, category_hierarchy, merchant cache, and the
"General Spending" / "Other Income" / "Investments" defaults).

Labeling is now up to US:

    1. `ml/bank_statment_processor.py` builds a clean, UNLABELED dataset
       at `all_banks_final_categorized.csv`. The `main_category` and
       `sub_category` columns are empty on purpose.
    2. We hand-label some rows in that CSV (this IS our training data).
       Picking our own category for every transaction is the "users
       select their own categories" part.
    3. We write the ML functions in this file, train them on those
       labeled rows, and then use the trained models to categorize
       EVERY INDIVIDUAL TRANSACTION automatically.

MANUAL WORKFLOW (what we do BY HAND)
------------------------------------
  STEP 1  Build the unlabeled dataset
            python -m ml.bank_statment_processor

  STEP 2  Label a batch of transactions
            Open `all_banks_final_categorized.csv` and fill in
            `main_category` and `sub_category` for as many rows as we
            can. (Split the file between us to go twice as fast.)

  STEP 3  Install the ML libraries (run in the project folder)
            uv add scikit-learn
            # optional, only if we later try real text embeddings:
            # uv add sentence-transformers

  STEP 4  Implement every TODO in this file. The scaffolding and data
            plumbing are written for us - the ML code is ours to write.

  STEP 5  Train on our labels, then categorize every transaction
            python -c "from ml.ml_models import train_and_categorize; train_and_categorize()"

  STEP 6  Evaluate
            Look at accuracy + the confusion matrix from
            evaluate_categorizer(). Which categories get confused?
            Which merchants show up in test but never in training?

CROSS-PERSON EXPERIMENT (our project question)
----------------------------------------------
"Can a model trained on one person's bank statements correctly
categorize the statements of another?"  To answer it we need a
`person` column (A or B) in the CSV - add it by hand, then run
cross_person_experiment() below.

BASELINES WE HAVE TO BEAT
-------------------------
  * always predict the most common category in training data
    (baseline_most_common())
  * per-merchant lookup: every merchant gets the category it appeared
    with most often in training
If our ML models can't beat those, something is wrong.

DATASET COLUMNS WE WORK WITH (output of bank_statement_processor.py)
--------------------------------------------------------------------
  date, description (cleaned), raw_description (original), type,
  amount, main_category (blank), sub_category (blank), bank, account
"""

import pandas as pd
import numpy as np

CSV_PATH = "all_banks_final_categorized.csv"  # produced by bank_statement_processor.py


# ---------------------------------------------------------------------------
# 0. DATA PLUMBING - already written, nothing to do here
# ---------------------------------------------------------------------------

def load_dataset(csv_path=CSV_PATH):
    """Read the processed CSV into a DataFrame with clean text columns."""
    df = pd.read_csv(csv_path)
    if "description" not in df.columns:
        raise ValueError(f"CSV at {csv_path} is missing a 'description' column.")
    df["description"] = df["description"].fillna("").astype(str)
    return df


def load_labeled(df):
    """Keep only the rows we have hand-labeled (main_category is filled in)."""
    return df.dropna(subset=["main_category"]).copy()


# ---------------------------------------------------------------------------
# 1. FEATURES - turn transaction text into numbers an ML model can learn from
# ---------------------------------------------------------------------------

def vectorize_descriptions(descriptions, max_features=5000):
    """TODO (we write this): text -> numeric feature matrix.

    What to implement:
        1. from sklearn.feature_extraction.text import TfidfVectorizer
        2. vectorizer = TfidfVectorizer(
               lowercase=True,
               ngram_range=(1, 2),      # single words + 2-word phrases
               max_features=max_features,
           )
        3. X = vectorizer.fit_transform(descriptions)
        4. return X, vectorizer

    Return the fitted vectorizer too, because we MUST reuse it to
    transform new transactions later (never fit again on new data).

    Extra features we can add once this works:
        - the sign of `amount` (negative = spending, positive = income)
        - `bank` / `account` type (checking vs savings vs credit card)
        - day of week / month from `date`
        - compare raw_description (original text) vs description
          (cleaned text) - that comparison is one of our questions
        - optional: pretrained embeddings (sentence-transformers)
          instead of tf-idf bag of words
    """
    raise NotImplementedError(
        "TODO: implement vectorize_descriptions() - steps in the docstring."
    )


# ---------------------------------------------------------------------------
# 2. MODELS - one training helper per ML method we compare
# ---------------------------------------------------------------------------

def train_knn(X_train, y_train, n_neighbors=5, metric="cosine"):
    """TODO (we write this): k-nearest-neighbors classifier.

    What to implement:
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
        model.fit(X_train, y_train)
        return model
    """
    raise NotImplementedError("TODO: implement train_knn()")


def train_random_forest(X_train, y_train, n_estimators=200, max_depth=None):
    """TODO (we write this): random forest classifier.

    What to implement:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        return model
    """
    raise NotImplementedError("TODO: implement train_random_forest()")


def train_neural_network(X_train, y_train, hidden_layer_sizes=(100, 50), max_iter=300):
    """TODO (we write this): ANN / multilayer perceptron classifier.

    What to implement:
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=42)
        model.fit(X_train, y_train)
        return model
    """
    raise NotImplementedError("TODO: implement train_neural_network()")


def train_categorizer(X_train, y_train, method="random_forest"):
    """TODO (we write this): wrapper that calls the right trainer above.

    method (str): one of "knn", "random_forest", "neural_network".
    We will train TWO separate models per method - one that predicts
    main_category and one that predicts sub_category - because those
    are two different label sets.
    """
    raise NotImplementedError("TODO: implement train_categorizer()")


# ---------------------------------------------------------------------------
# 3. CATEGORIZE - the part that labels each INDIVIDUAL transaction
# ---------------------------------------------------------------------------

def categorize_transaction(description, amount=None,
                           vectorizer=None, main_model=None, sub_model=None):
    """TODO (we write this): predict categories for ONE transaction.

    This is the function that assigns a main_category / sub_category to
    a single transaction - the core of what we want the ML to do.

    What to implement:
        1. X_row = vectorizer.transform([str(description)])
           # if we added amount/bank as features, build that row too
        2. main = main_model.predict(X_row)[0]
        3. sub  = sub_model.predict(X_row)[0]
        4. return {"main_category": main, "sub_category": sub}
    """
    raise NotImplementedError("TODO: implement categorize_transaction()")


def categorize_dataframe(df, vectorizer, main_model, sub_model):
    """TODO (we write this): fill category columns for EVERY row in df.

    Loop over all rows (df.itertuples() or df.apply), call
    categorize_transaction() on each one, and write the predictions back
    into the DataFrame. Return the updated df.
    """
    raise NotImplementedError("TODO: implement categorize_dataframe()")


def evaluate_categorizer(y_true, y_pred, label_name="main_category"):
    """TODO (we write this): accuracy + per-category + confusion matrix.

    What to implement:
        from sklearn.metrics import (accuracy_score, classification_report,
                                     confusion_matrix)
    Print/report:
        - overall accuracy
        - accuracy per category - small categories matter
          (e.g. "Fast Offerings", "CC Rewards")
        - a confusion matrix so we can study which categories get mixed up
    """
    raise NotImplementedError("TODO: implement evaluate_categorizer()")


# ---------------------------------------------------------------------------
# 4. END-TO-END RUNNERS - train on our labels, then categorize everything
# ---------------------------------------------------------------------------

def train_and_categorize(csv_path=CSV_PATH,
                         output_csv="all_banks_final_ml_categorized.csv",
                         method="random_forest"):
    """TODO (we write this): the whole pipeline in one call.

    Suggested steps:
        1. df  = load_dataset(csv_path)
        2. lab = load_labeled(df)
        3. X, vectorizer = vectorize_descriptions(lab["description"])
        4. Split FIRST (hold out a test set so we can honestly evaluate):
               from sklearn.model_selection import train_test_split
               X_tr, X_te, y_tr_main, y_te_main = train_test_split(
                   X, lab["main_category"], test_size=0.2, random_state=42,
                   stratify=lab["main_category"])
        5. main_model = train_categorizer(X_tr, y_tr_main, method=method)
        6. sub_model  = train_categorizer(
               X_tr, y_tr_sub, method=method)
           # y_tr_sub = the sub_category labels for the same training rows
        7. evaluate both models on X_te (see evaluate_categorizer())
        8. full = categorize_dataframe(df, vectorizer, main_model, sub_model)
           # now EVERY transaction - labeled or not - gets a prediction
        9. full.to_csv(output_csv, index=False)

    MANUAL COMMAND (after implementing):
        python -c "from ml.ml_models import train_and_categorize; train_and_categorize()"
    """
    raise NotImplementedError("TODO: implement train_and_categorize()")


def baseline_most_common(df, target="main_category"):
    """TODO (we write this): the most boring baseline.

    Always predict the most frequent label in the training rows and
    return the accuracy it gets on held-out rows. If our ML can't beat
    this, we know something is off.
    """
    raise NotImplementedError("TODO: implement baseline_most_common()")


def cross_person_experiment(csv_path=CSV_PATH):
    """TODO (we write this): train on person A, test on person B.

    Only possible after we add a `person` column (A or B) to the CSV.

    Do BOTH directions:
        - train on A's labeled rows, evaluate on B's labeled rows
        - train on B's labeled rows, evaluate on A's labeled rows
    Compare each against baseline_most_common() and against training +
    testing within the same person. This answers our project question.
    """
    raise NotImplementedError("TODO: implement cross_person_experiment()")


if __name__ == "__main__":
    print(__doc__)