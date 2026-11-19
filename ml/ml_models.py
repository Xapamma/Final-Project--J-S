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

    1. `ml/bank_statement_processor.py` builds one clean, UNLABELED dataset
       per person. The `main_category` and `sub_category` columns are empty
       on purpose.
    2. We hand-label some rows in that CSV (this IS our training data).
       Picking our own category for every transaction is the "users
       select their own categories" part.
    3. We write the ML functions in this file, train them on those
       labeled rows, and then use the trained models to categorize
       EVERY INDIVIDUAL TRANSACTION automatically.

REPRODUCIBLE WORKFLOW
---------------------
  STEP 1  Build the unlabeled dataset
            uv run python -m ml.bank_statment_processor

  STEP 2  Hand-label `main_category` and `sub_category` in the
            generated CSV for as many rows as possible.

  STEP 3  Install project dependencies
            uv sync

  STEP 4  Train and categorize transactions
            uv run python -c "from ml.ml_models import train_and_categorize; train_and_categorize()"

  STEP 5  Evaluate the original fixed baselines
            uv run python -m ml.first_result --savanna savanna_transactions.csv
            uv run python -m ml.first_result --jillian jillian_transactions_labeled.csv

  STEP 6  Compare tuned models and amount multipliers
            uv run python -m ml.model_comparison

CROSS-PERSON EXPERIMENT (our project question)
----------------------------------------------
"Can a model trained on one person's bank statements correctly
categorize the statements of another?"  To answer it we need a
`person` column (A or B) in the CSV - add it by hand, then run
cross_person_experiment() below.

MODULE ORGANIZATION
-------------------
  * `ml_models.py` owns reusable data plumbing, feature construction,
    model trainers, parameter grids, tuning helpers, and evaluation.
  * `first_result.py` reports the unchanged fixed 5-NN baseline.
  * `model_comparison.py` performs outer-fold model comparison.

BASELINES AND COMPARISONS
-------------------------
  * Always predict the most common training category
    (`baseline_most_common`).
  * Character TF-IDF KNN and sentence-embedding KNN.
  * Random forest and neural network on embedding features.
  * Tuned KNN comparisons also select the amount multiplier and KNN
    parameters with cross-validation.

DATASET COLUMNS WE WORK WITH (output of bank_statement_processor.py)
--------------------------------------------------------------------
  date, description (original bank text), type,
  amount, main_category (blank), sub_category (blank), bank, account
"""

from functools import lru_cache

import numpy as np
import pandas as pd

CSV_PATH = "savanna_transactions.csv"  # produced by bank_statment_processor.py


# ---------------------------------------------------------------------------
# 0. DATA PLUMBING - already written, nothing to do here
# ---------------------------------------------------------------------------

def load_dataset(csv_path=CSV_PATH):
    """Read the processed CSV into a DataFrame with clean text columns."""
    # keep_default_na=False: "N/A" is a real sub-category (Transfer / N/A),
    # not a missing value. Only truly empty cells count as unlabeled.
    df = pd.read_csv(csv_path, keep_default_na=False, na_values=[""])
    if "description" not in df.columns and "raw_description" in df.columns:
        # Older processor output kept the bank text in `raw_description`.
        df = df.rename(columns={"raw_description": "description"})
    if "description" not in df.columns:
        raise ValueError(f"CSV at {csv_path} is missing a 'description' column.")
    df["description"] = df["description"].fillna("").astype(str)
    return df


def load_labeled(df):
    """Keep only the rows we have hand-labeled (main_category is filled in)."""
    return df.dropna(subset=["main_category"]).copy()


# ---------------------------------------------------------------------------
# 1. FEATURES - turn transaction text and amounts into model features
# ---------------------------------------------------------------------------

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_AMOUNT_WEIGHT = 25.0


@lru_cache(maxsize=4)
def _load_embedding_model(model_name):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_descriptions(descriptions, model_name=DEFAULT_EMBEDDING_MODEL):
    """Encode descriptions with a pretrained sentence-embedding model.

    The model is loaded lazily and is not fitted on transaction data. This
    keeps the text representation consistent across folds and people while
    avoiding label or test-set leakage.
    """
    model = _load_embedding_model(model_name)
    values = pd.Series(descriptions).fillna("").astype(str).tolist()
    embeddings = model.encode(
        values,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings, model


def vectorize_descriptions(descriptions, max_features=5000, analyzer="char_wb"):
    """Create train-fitted character TF-IDF features for comparison KNN."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    ngram_range = (3, 5) if analyzer in ("char", "char_wb") else (1, 2)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer=analyzer,
        ngram_range=ngram_range,
        max_features=max_features,
    )
    values = pd.Series(descriptions).fillna("").astype(str)
    return vectorizer.fit_transform(values), vectorizer


def _amount_features(train_amounts, test_amounts, weight=DEFAULT_AMOUNT_WEIGHT):
    """Scale amounts using training statistics, then apply their KNN weight."""
    train_values = pd.to_numeric(train_amounts, errors="coerce").fillna(0.0)
    test_values = pd.to_numeric(test_amounts, errors="coerce").fillna(0.0)
    mean = float(train_values.mean())
    scale = float(train_values.std(ddof=0))
    if scale == 0:
        scale = 1.0
    train_scaled = ((train_values - mean) / scale).to_numpy().reshape(-1, 1)
    test_scaled = ((test_values - mean) / scale).to_numpy().reshape(-1, 1)
    return train_scaled * weight, test_scaled * weight


def build_embedding_features(
    train,
    test,
    *,
    amount_weight=DEFAULT_AMOUNT_WEIGHT,
    model_name=DEFAULT_EMBEDDING_MODEL,
):
    """Build leakage-safe description-plus-amount features for KNN.

    Amounts are standardized from the training rows only before being
    multiplied by ``amount_weight`` and concatenated to the text embeddings.
    """
    train_embeddings, model = embed_descriptions(train["description"], model_name)
    test_embeddings = model.encode(
        test["description"].fillna("").astype(str).tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    train_amounts, test_amounts = _amount_features(
        train["amount"], test["amount"], weight=amount_weight
    )
    return (
        np.hstack([train_embeddings, train_amounts]),
        np.hstack([test_embeddings, test_amounts]),
        model,
    )


def build_tfidf_features(
    train,
    test,
    *,
    amount_weight=DEFAULT_AMOUNT_WEIGHT,
):
    """Build TF-IDF-plus-amount features using training data only."""
    from scipy.sparse import hstack

    train_text, vectorizer = vectorize_descriptions(train["description"])
    test_text = vectorizer.transform(test["description"].fillna("").astype(str))
    train_amounts, test_amounts = _amount_features(
        train["amount"], test["amount"], weight=amount_weight
    )
    return (
        hstack([train_text, train_amounts]).tocsr(),
        hstack([test_text, test_amounts]).tocsr(),
    )


def oversample_small_categories(X, y, minimum_support=20, random_state=42):
    """Oversample categories below a support threshold inside training data only."""
    y = pd.Series(y).reset_index(drop=True)
    rng = np.random.default_rng(random_state)
    counts = y.value_counts()
    selected_indices = np.arange(len(y), dtype=int).tolist()

    for category, count in counts.items():
        if count >= minimum_support:
            continue
        category_indices = y.index[y == category].to_numpy(dtype=int)
        needed = minimum_support - len(category_indices)
        selected_indices.extend(
            rng.choice(category_indices, size=needed, replace=True).tolist()
        )

    indices = np.asarray(selected_indices, dtype=int)
    from scipy.sparse import issparse, vstack

    X_resampled = vstack([X[index:index + 1] for index in indices]) if issparse(X) else X[indices]
    y_resampled = y.iloc[indices].reset_index(drop=True)
    return X_resampled, y_resampled


def build_embedding_feature_ablation(
    train,
    test,
    *,
    include_amount=False,
    include_type=False,
    include_account=False,
    amount_weight=1.0,
    model_name=DEFAULT_EMBEDDING_MODEL,
):
    """Build embedding features with leakage-safe optional tabular features."""
    train_embeddings, model = embed_descriptions(
        train["description"], model_name
    )
    test_embeddings = model.encode(
        test["description"].fillna("").astype(str).tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    train_parts = [train_embeddings]
    test_parts = [test_embeddings]

    if include_amount:
        train_amount, test_amount = _amount_features(
            train["amount"],
            test["amount"],
            weight=amount_weight,
        )
        train_parts.append(train_amount)
        test_parts.append(test_amount)

    for column in ("type", "account"):
        include = include_type if column == "type" else include_account
        if not include:
            continue
        train_values = pd.Series(train[column].fillna("Unknown").astype(str).str.strip())
        test_values = pd.Series(test[column].fillna("Unknown").astype(str).str.strip())
        train_categories = sorted(train_values.unique())
        train_dummies = pd.get_dummies(
            train_values,
            prefix=column,
            dtype=float,
        ).reindex(columns=[f"{column}_{value}" for value in train_categories], fill_value=0.0)
        test_dummies = pd.get_dummies(
            test_values,
            prefix=column,
            dtype=float,
        ).reindex(columns=train_dummies.columns, fill_value=0.0)
        train_parts.append(train_dummies.to_numpy(dtype=float))
        test_parts.append(test_dummies.to_numpy(dtype=float))

    return (
        np.hstack(train_parts),
        np.hstack(test_parts),
        model,
    )


# ---------------------------------------------------------------------------
# 2. MODELS - one training helper per ML method we compare
# ---------------------------------------------------------------------------

def train_knn(X_train, y_train, n_neighbors=5, metric="cosine"):
    """Train KNN on the selected text-and-amount features."""
    from sklearn.neighbors import KNeighborsClassifier

    model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators=200, max_depth=None):
    """Train a random forest classifier."""
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    return model


def train_neural_network(
    X_train,
    y_train,
    hidden_layer_sizes=(100,),
    max_iter=1000,
    alpha=0.0001,
    learning_rate_init=0.001,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20,
):
    """Train an MLP (ANN) classifier."""
    import warnings

    from sklearn.neural_network import MLPClassifier
    from sklearn.exceptions import ConvergenceWarning

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=42,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        n_iter_no_change=n_iter_no_change,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        model.fit(X_train, y_train)
    return model


def train_categorizer(X_train, y_train, method="random_forest", **params):
    """Train the requested classifier.

    ``params`` are forwarded to the corresponding trainer, for example
    ``n_neighbors=7`` for KNN or ``n_estimators=400`` for a random forest.
    """
    trainers = {
        "knn": train_knn,
        "random_forest": train_random_forest,
        "neural_network": train_neural_network,
    }
    if method not in trainers:
        raise ValueError(
            f"Unknown method {method!r}; choose from {sorted(trainers)}."
        )
    return trainers[method](X_train, y_train, **params)


# Parameter grids used for cross-validation and hyperparameter tuning.
CV_PARAM_GRIDS = {
    "knn": {
        "n_neighbors": [3, 5, 7, 9],
        "metric": ["cosine", "euclidean"],
    },
    "random_forest": {
        "n_estimators": [100, 200, 400],
        "max_depth": [None, 10, 20],
    },
    "neural_network": {
        "hidden_layer_sizes": [(50,), (100,)],
        "max_iter": [500, 1000],
        "alpha": [0.00001, 0.0001, 0.001, 0.01],
        "learning_rate_init": [0.001],
        "early_stopping": [True],
        "validation_fraction": [0.15],
        "n_iter_no_change": [20],
    },
}


def tune_model(X, y, estimator, param_grid, cv=5, scoring="f1_macro"):
    """Cross-validate estimator parameters and return the fitted search.

    The returned ``GridSearchCV`` is fitted and exposes ``best_estimator_``,
    ``best_params_``, ``best_score_``, and ``cv_results_``.
    """
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    y_counts = pd.Series(y).value_counts()
    usable_splits = int(y_counts.min()) if len(y_counts) else 0
    if usable_splits < 2:
        raise ValueError(
            "Cross-validation tuning requires at least 2 rows in every category."
        )
    n_splits = min(cv, usable_splits)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    search = GridSearchCV(
        estimator,
        param_grid,
        cv=splitter,
        scoring=scoring,
        n_jobs=1,
        error_score=0.0,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        search.fit(X, y)
    return search


def tune_categorizer(X, y, method="random_forest", param_grid=None, cv=5):
    """Tune one categorizer and return its fitted GridSearchCV search."""
    if method == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        estimator = KNeighborsClassifier()
    elif method == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        estimator = RandomForestClassifier(random_state=42, n_jobs=1)
    elif method == "neural_network":
        from sklearn.neural_network import MLPClassifier
        estimator = MLPClassifier(
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
        )
    else:
        raise ValueError(
            "Unknown method "
            f"{method!r}; choose from ['knn', 'random_forest', 'neural_network']. "
        )
    grid = param_grid if param_grid is not None else CV_PARAM_GRIDS[method]
    return tune_model(X, y, estimator, grid, cv=cv)


def tune_knn(X, y, param_grid=None, cv=5):
    return tune_categorizer(X, y, "knn", param_grid=param_grid, cv=cv)


def tune_random_forest(X, y, param_grid=None, cv=5):
    return tune_categorizer(X, y, "random_forest", param_grid=param_grid, cv=cv)


def tune_neural_network(X, y, param_grid=None, cv=5):
    return tune_categorizer(X, y, "neural_network", param_grid=param_grid, cv=cv)


AMOUNT_WEIGHT_GRID = [0.0, 1.0]


def tune_amount_multiplier(
    train,
    target="main_category",
    feature_type="embedding",
    amount_weights=None,
    knn_param_grid=None,
    cv=3,
    oversampling_threshold=20,
):
    """Tune the amount multiplier together with KNN parameters.

    Inner splits use stratified rows within ``train``. For every multiplier
    and KNN parameter combination, text and amount features are rebuilt for
    the inner training/validation split so amount scaling and TF-IDF fitting
    remain leakage-safe.

    Returns ``(best_result, cv_results)``. ``best_result`` contains
    ``amount_weight``, ``n_neighbors``, ``metric``, and ``mean_macro_f1``.
    """
    from itertools import product

    from sklearn.model_selection import StratifiedKFold

    if feature_type not in ("embedding", "tfidf"):
        raise ValueError(
            f"Unknown feature_type {feature_type!r}; use 'embedding' or 'tfidf'."
        )

    train = load_labeled(train).reset_index(drop=True)
    y = train[target]
    counts = y.value_counts()
    if len(counts) < 2 or int(counts.min()) < 2:
        return {
            "amount_weight": 1.0,
            "n_neighbors": 5,
            "metric": "cosine",
            "mean_macro_f1": None,
        }, pd.DataFrame()

    weights = amount_weights or AMOUNT_WEIGHT_GRID
    grid = knn_param_grid or CV_PARAM_GRIDS["knn"]
    neighbors = grid.get("n_neighbors", [5])
    metrics = grid.get("metric", ["cosine"])
    n_splits = min(cv, int(counts.min()))
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    rows = []
    combinations = list(product(weights, neighbors, metrics))
    for amount_weight, n_neighbors, metric in combinations:
        fold_scores = []
        for inner_train_idx, inner_test_idx in splitter.split(
            train,
            train[target],
        ):
            inner_train = train.iloc[inner_train_idx]
            inner_test = train.iloc[inner_test_idx]
            if feature_type == "embedding":
                X_inner_train, X_inner_test, _ = build_embedding_features(
                    inner_train,
                    inner_test,
                    amount_weight=amount_weight,
                )
            else:
                X_inner_train, X_inner_test = build_tfidf_features(
                    inner_train,
                    inner_test,
                    amount_weight=amount_weight,
                )
            X_inner_train, inner_y = oversample_small_categories(
                X_inner_train,
                inner_train[target],
                minimum_support=oversampling_threshold,
                random_state=42,
            )
            model = train_knn(
                X_inner_train,
                inner_y,
                n_neighbors=n_neighbors,
                metric=metric,
            )
            pred = model.predict(X_inner_test)
            score = evaluate_categorizer(
                inner_test[target],
                pred,
                verbose=False,
            )
            fold_scores.append(score["macro_f1"])
        rows.append({
            "amount_weight": amount_weight,
            "n_neighbors": n_neighbors,
            "metric": metric,
            "mean_macro_f1": float(np.mean(fold_scores)),
        })

    cv_results = pd.DataFrame(rows)
    best_row = cv_results.sort_values(
        ["mean_macro_f1", "amount_weight", "n_neighbors", "metric"],
        ascending=[False, True, True, True],
    ).iloc[0]
    best_result = {
        "amount_weight": float(best_row["amount_weight"]),
        "n_neighbors": int(best_row["n_neighbors"]),
        "metric": str(best_row["metric"]),
        "mean_macro_f1": float(best_row["mean_macro_f1"]),
    }
    return best_result, cv_results


# ---------------------------------------------------------------------------
# 3. CATEGORIZE - the part that labels each INDIVIDUAL transaction
# ---------------------------------------------------------------------------

def categorize_transaction(description, amount=None,
                           vectorizer=None, main_model=None, sub_model=None):
    """Predict main and sub categories for one transaction."""
    if main_model is None or sub_model is None:
        raise ValueError("main_model and sub_model are required.")

    row = pd.DataFrame({
        "description": ["" if description is None else str(description)],
        "amount": [0.0 if amount is None else amount],
    })
    # Embedding-trained models use the lazy shared embedding helper.
    # TF-IDF-trained models can supply vectorizer and use it here too.
    if vectorizer is not None:
        text = vectorizer.transform(row["description"].fillna(""))
        from scipy.sparse import hstack
        amount_value = pd.to_numeric(row["amount"], errors="coerce").fillna(0.0)
        mean = getattr(main_model, "_amount_mean", 0.0)
        scale = getattr(main_model, "_amount_scale", 1.0)
        weight = getattr(main_model, "_amount_weight", DEFAULT_AMOUNT_WEIGHT)
        scaled_amount = ((amount_value - mean) / scale).to_numpy().reshape(-1, 1)
        X_row = hstack([text, scaled_amount * weight]).tocsr()
    else:
        embeddings, model = embed_descriptions(row["description"])
        amount_value = pd.to_numeric(row["amount"], errors="coerce").fillna(0.0)
        mean = getattr(main_model, "_amount_mean", 0.0)
        scale = getattr(main_model, "_amount_scale", 1.0)
        weight = getattr(main_model, "_amount_weight", DEFAULT_AMOUNT_WEIGHT)
        scaled_amount = ((amount_value - mean) / scale).to_numpy().reshape(-1, 1)
        X_row = np.hstack([embeddings, scaled_amount * weight])

    main = main_model.predict(X_row)[0]
    sub = sub_model.predict(X_row)[0]
    return {"main_category": main, "sub_category": sub}


def categorize_dataframe(df, vectorizer, main_model, sub_model):
    """Fill category columns for every row in a DataFrame."""
    result = df.copy()
    predictions = [
        categorize_transaction(
            row.description,
            getattr(row, "amount", None),
            vectorizer=vectorizer,
            main_model=main_model,
            sub_model=sub_model,
        )
        for row in result.itertuples(index=False)
    ]
    result["main_category"] = [item["main_category"] for item in predictions]
    result["sub_category"] = [item["sub_category"] for item in predictions]
    return result


def evaluate_categorizer(y_true, y_pred, label_name="main_category", verbose=True):
    """Accuracy, macro-F1, per-category report, and confusion matrix.

    Macro-F1 averages F1 over categories equally, so small categories
    (e.g. "Fast Offerings", "CC Rewards") count as much as big ones -
    important because our categories are very imbalanced.

    Returns a dict with:
        accuracy, macro_f1      floats
        report                  DataFrame, one row per category
                                (precision, recall, f1-score, support)
        confusion_matrix        DataFrame, rows = true, columns = predicted
    """
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score)

    y_true = pd.Series(y_true, dtype="string").reset_index(drop=True)
    y_pred = pd.Series(y_pred, dtype="string").reset_index(drop=True)
    labels = sorted(set(y_true) | set(y_pred))

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    report = pd.DataFrame(
        classification_report(y_true, y_pred, labels=labels,
                              output_dict=True, zero_division=0)
    ).T.loc[labels]
    report["support"] = report["support"].astype(int)
    matrix = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=labels),
        index=pd.Index(labels, name="true"),
        columns=pd.Index(labels, name="predicted"),
    )

    if verbose:
        print(f"=== {label_name} ===")
        print(f"accuracy: {accuracy:.3f}   macro-F1: {macro_f1:.3f}   (n={len(y_true)})")
        print(report.round(3).to_string())
        print("\nconfusion matrix (rows = true, columns = predicted):")
        print(matrix.to_string())

    return {"accuracy": accuracy, "macro_f1": macro_f1,
            "report": report, "confusion_matrix": matrix}


def within_person_cv(df, fit_predict, target="main_category", n_splits=5,
                     group_col="description"):
    """Within-person evaluation with GroupKFold grouped by description.

    WHY GROUPS: the same description shows up many times (e.g. 49 "Chevron"
    rows). With a plain random split, copies of a description land in both
    train and test and the model just memorizes it - accuracy looks great
    but says nothing about NEW descriptions. Grouping by description keeps
    every copy of a description in the same fold.

    fit_predict(train_df, test_df) -> predictions for test_df.
        Do all train-data-dependent preparation inside it using only
        train_df, so nothing about the test fold leaks into training.
        Example with our functions:
            def knn_fit_predict(train, test):
                X_tr, X_te, _ = build_embedding_features(train, test)
                model = train_knn(X_tr, train["main_category"])
                return model.predict(X_te)

    Returns (fold_scores DataFrame, overall result from evaluate_categorizer
    computed on the pooled out-of-fold predictions).
    """
    from sklearn.model_selection import GroupKFold

    labeled = load_labeled(df).reset_index(drop=True)
    groups = labeled[group_col].astype(str).str.strip().str.lower()
    n_groups = groups.nunique()
    if n_groups < n_splits:
        raise ValueError(f"Only {n_groups} distinct merchants; need at least n_splits={n_splits}.")

    predictions = pd.Series(pd.NA, index=labeled.index, dtype="string")
    fold_rows = []
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(labeled, groups=groups), start=1):
        train, test = labeled.iloc[train_idx], labeled.iloc[test_idx]
        # Sanity check: no merchant in both train and test.
        assert not set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
        fold_pred = pd.Series(list(fit_predict(train, test)), index=test.index, dtype="string")
        predictions.loc[test.index] = fold_pred
        scores = evaluate_categorizer(test[target], fold_pred, verbose=False)
        fold_rows.append({"fold": fold, "n_test": len(test),
                          "accuracy": scores["accuracy"], "macro_f1": scores["macro_f1"]})

    fold_scores = pd.DataFrame(fold_rows)
    overall = evaluate_categorizer(labeled[target], predictions, label_name=target, verbose=False)
    return fold_scores, overall


# ---------------------------------------------------------------------------
# 4. END-TO-END RUNNERS - train on our labels, then categorize everything
# ---------------------------------------------------------------------------

def train_and_categorize(csv_path=CSV_PATH,
                         output_csv="all_banks_final_ml_categorized.csv",
                         method="random_forest", tune=False, cv=5,
                         amount_weight=DEFAULT_AMOUNT_WEIGHT,
                         model_name=DEFAULT_EMBEDDING_MODEL):
    """Train main/sub models, evaluate a holdout, and categorize every row."""
    from sklearn.model_selection import train_test_split

    df = load_dataset(csv_path)
    lab = load_labeled(df).reset_index(drop=True)
    if len(lab) < 2:
        raise ValueError("At least two labeled rows are required.")

    main_counts = lab["main_category"].value_counts()
    can_stratify = bool((main_counts >= 2).all()) and len(lab) >= 4
    train, test = train_test_split(
        lab,
        test_size=0.2,
        random_state=42,
        stratify=lab["main_category"] if can_stratify else None,
    )

    X_train, X_test, _ = build_embedding_features(
        train,
        test,
        amount_weight=amount_weight,
        model_name=model_name,
    )

    if tune:
        main_counts = train["main_category"].value_counts()
        if len(main_counts) >= 2 and int(main_counts.min()) >= 2:
            main_search = tune_categorizer(
                X_train, train["main_category"], method=method, cv=cv
            )
            main_model = main_search.best_estimator_
        else:
            main_model = train_categorizer(
                X_train, train["main_category"], method=method
            )

        sub_train_counts = train["sub_category"].fillna("Unknown").value_counts()
        sub_safe = sub_train_counts[sub_train_counts >= 2].index
        sub_train = train["sub_category"].fillna("Unknown").isin(sub_safe)
        X_sub_train = X_train[sub_train.to_numpy()]
        y_sub_train = train.loc[sub_train, "sub_category"].fillna("Unknown")
        if y_sub_train.nunique() >= 2:
            sub_search = tune_categorizer(
                X_sub_train, y_sub_train, method=method, cv=cv
            )
            sub_model = sub_search.best_estimator_
        else:
            sub_model = train_categorizer(
                X_sub_train, y_sub_train, method=method
            )
    else:
        main_model = train_categorizer(
            X_train, train["main_category"], method=method
        )
        sub_model = train_categorizer(
            X_train,
            train["sub_category"].fillna("Unknown"),
            method=method,
        )

    main_pred = main_model.predict(X_test)
    sub_pred = sub_model.predict(
        X_test
        if hasattr(sub_model, "predict")
        else X_test
    )
    evaluate_categorizer(
        test["main_category"], main_pred, label_name="main_category"
    )
    evaluate_categorizer(
        test["sub_category"].fillna("Unknown"),
        sub_pred,
        label_name="sub_category",
    )

    train_values = pd.to_numeric(train["amount"], errors="coerce").fillna(0.0)
    mean = float(train_values.mean())
    scale = float(train_values.std(ddof=0))
    if scale == 0:
        scale = 1.0
    for model in (main_model, sub_model):
        model._amount_mean = mean
        model._amount_scale = scale
        model._amount_weight = amount_weight

    full = categorize_dataframe(df, None, main_model, sub_model)
    full.to_csv(output_csv, index=False)
    return {
        "main_model": main_model,
        "sub_model": sub_model,
        "output_csv": output_csv,
        "main_accuracy": float((test["main_category"].astype(str).to_numpy()
                                == pd.Series(main_pred).astype(str).to_numpy()).mean()),
    }


# ---------------------------------------------------------------------------
# 5. FIRST RESULT - baseline + KNN as fit_predict functions
#    Each takes (train_df, test_df) and returns predictions for test_df.
#    All fitting uses train_df only, so they plug straight into
#    within_person_cv() and cross_person_experiment() without leakage.
# ---------------------------------------------------------------------------

def baseline_most_common(train, test, target="main_category"):
    """Always predict the most frequent label in the training rows."""
    return [train[target].value_counts().idxmax()] * len(test)


def knn_embeddings(
    train,
    test,
    target="main_category",
    n_neighbors=5,
    amount_weight=DEFAULT_AMOUNT_WEIGHT,
    model_name=DEFAULT_EMBEDDING_MODEL,
):
    """Predict with sentence embeddings plus a weighted standardized amount."""
    X_train, X_test, _ = build_embedding_features(
        train,
        test,
        amount_weight=amount_weight,
        model_name=model_name,
    )
    model = train_knn(X_train, train[target], n_neighbors=n_neighbors)
    return model.predict(X_test)


def knn_tfidf(
    train,
    test,
    target="main_category",
    n_neighbors=5,
    amount_weight=DEFAULT_AMOUNT_WEIGHT,
):
    """Predict with TF-IDF character features plus a weighted amount."""
    X_train, X_test = build_tfidf_features(
        train,
        test,
        amount_weight=amount_weight,
    )
    model = train_knn(X_train, train[target], n_neighbors=n_neighbors)
    return model.predict(X_test)


FIRST_RESULT_MODELS = {
    "most_common": baseline_most_common,
    "knn_tfidf": knn_tfidf,
    "knn_embeddings": knn_embeddings,
}


def cross_person_experiment(train_df, test_df, models=None, target="main_category"):
    """Train on one person's labeled rows, evaluate on the other's.

    Call it both ways (A->B and B->A). Returns one row per model with
    accuracy and macro-F1, plus the share of test rows whose exact
    description also appears in training (familiar descriptions).
    """
    models = models or FIRST_RESULT_MODELS
    train, test = load_labeled(train_df), load_labeled(test_df)
    seen = test["description"].str.lower().isin(set(train["description"].str.lower()))
    rows = []
    for name, fit_predict in models.items():
        scores = evaluate_categorizer(test[target], fit_predict(train, test), verbose=False)
        rows.append({"model": name, "accuracy": scores["accuracy"],
                     "macro_f1": scores["macro_f1"], "n_test": len(test),
                     "familiar_descriptions": seen.mean()})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(__doc__)
