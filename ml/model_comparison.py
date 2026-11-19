"""Cross-validated comparison for the KNN, random forest, and neural network models.

Run from the project root:
    uv run python -m ml.model_comparison

The original first-result runner and fixed 5-NN models are not changed.
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold, ParameterGrid, StratifiedKFold

from ml.ml_models import (
    CV_PARAM_GRIDS,
    baseline_most_common,
    build_embedding_feature_ablation,
    build_tfidf_features,
    evaluate_categorizer,
    load_dataset,
    load_labeled,
    oversample_small_categories,
    train_categorizer,
)


def _feature_candidates(method):
    """Return candidate feature sets for one model family."""
    if method == "knn_tfidf":
        return [
            ("description", {"amount_weight": 0.0}),
            ("description + amount=1", {"amount_weight": 1.0}),
        ]
    return [
        ("description", {"include_amount": False, "include_type": False,
                         "include_account": False, "amount_weight": 0.0}),
        ("description + amount=1", {"include_amount": True, "include_type": False,
                                    "include_account": False, "amount_weight": 1.0}),
        ("description + type", {"include_amount": False, "include_type": True,
                                "include_account": False, "amount_weight": 0.0}),
        ("description + amount=1 + type", {"include_amount": True, "include_type": True,
                                            "include_account": False, "amount_weight": 1.0}),
        ("description + account",
         {"include_amount": False, "include_type": False, "include_account": True,
          "amount_weight": 0.0}),
        ("description + amount=1 + type + account",
         {"include_amount": True, "include_type": True, "include_account": True,
          "amount_weight": 1.0}),
    ]


def _build_features(method, train, test, options):
    if method == "knn_tfidf":
        return build_tfidf_features(train, test, **options)
    options = dict(options)
    options["include_type"] = options["include_type"] and "type" in train
    options["include_account"] = options["include_account"] and "account" in train
    return build_embedding_feature_ablation(train, test, **options)[:2]


def _model_method(method):
    return "knn" if method in ("knn_tfidf", "knn_embeddings") else method


def _default_model_params(method):
    if method in ("knn_tfidf", "knn_embeddings"):
        return {"n_neighbors": 5, "metric": "cosine"}
    if method == "random_forest":
        return {"n_estimators": 200, "max_depth": None}
    return {
        "hidden_layer_sizes": (100,), "max_iter": 1000, "alpha": 0.0001,
        "learning_rate_init": 0.001, "early_stopping": True,
        "validation_fraction": 0.15, "n_iter_no_change": 20,
    }


def _fit_tuned_model(train, test, method, options, cv=3, oversampling_threshold=20):
    """Tune with preprocessing and oversampling fitted inside each inner split."""
    y = train["main_category"]
    counts = y.value_counts()
    params = _default_model_params(method)
    metadata = {"status": "default: rare category"}
    if len(counts) >= 2 and int(counts.min()) >= 2:
        splitter = StratifiedKFold(n_splits=min(cv, int(counts.min())),
                                   shuffle=True, random_state=42)
        folds = []
        for tr_idx, val_idx in splitter.split(train, y):
            tr, val = train.iloc[tr_idx], train.iloc[val_idx]
            X_tr, X_val = _build_features(method, tr, val, options)
            X_tr, y_tr = oversample_small_categories(
                X_tr, tr["main_category"], minimum_support=oversampling_threshold,
                random_state=42)
            folds.append((X_tr, y_tr, X_val, val["main_category"]))
        candidates = []
        for candidate in ParameterGrid(CV_PARAM_GRIDS[_model_method(method)]):
            scores = []
            for X_tr, y_tr, X_val, y_val in folds:
                model = train_categorizer(X_tr, y_tr, method=_model_method(method), **candidate)
                prediction = model.predict(X_val)
                scores.append(evaluate_categorizer(y_val, prediction, verbose=False)["macro_f1"])
            candidates.append((sum(scores) / len(scores), candidate))
        best_score, params = max(candidates, key=lambda item: item[0])
        metadata = {"status": "tuned", "inner_cv_macro_f1": best_score}
    X_train, X_test = _build_features(method, train, test, options)
    X_train, y_train = oversample_small_categories(
        X_train, y, minimum_support=oversampling_threshold, random_state=42)
    model = train_categorizer(X_train, y_train, method=_model_method(method), **params)
    return model, {**params, **metadata}, X_test


def _select_feature_variant(train, method, cv=3, oversampling_threshold=20):
    """Select features using only stratified inner folds of one outer train set."""
    y = train["main_category"]
    counts = y.value_counts()
    if len(counts) < 2 or int(counts.min()) < 2:
        return _feature_candidates(method)[0]
    splitter = StratifiedKFold(n_splits=min(cv, int(counts.min())),
                               shuffle=True, random_state=42)
    scores = []
    for name, options in _feature_candidates(method):
        fold_scores = []
        for inner_train_idx, inner_test_idx in splitter.split(train, y):
            inner_train = train.iloc[inner_train_idx]
            inner_test = train.iloc[inner_test_idx]
            X_train, X_test = _build_features(method, inner_train, inner_test, options)
            X_train, y_train = oversample_small_categories(
                X_train, inner_train["main_category"],
                minimum_support=oversampling_threshold, random_state=42)
            default_params = _default_model_params(method)
            model = train_categorizer(
                X_train,
                y_train,
                method=_model_method(method),
                **default_params,
            )
            fold_scores.append(evaluate_categorizer(
                inner_test["main_category"], model.predict(X_test), verbose=False
            )["macro_f1"])
        scores.append((float(sum(fold_scores) / len(fold_scores)), name, options))
    return max(scores, key=lambda item: (item[0], item[1]))[1:]


def feature_ablation_within_person(df, n_splits=5, oversampling_threshold=20):
    """Evaluate every feature candidate separately for every model family."""
    labeled = load_labeled(df).reset_index(drop=True)
    groups = labeled["description"].astype(str).str.strip().str.lower()
    rows = []
    for method in ("knn_tfidf", "knn_embeddings", "random_forest", "neural_network"):
        for variant_name, options in _feature_candidates(method):
            predictions = pd.Series(pd.NA, index=labeled.index, dtype="string")
            for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(
                labeled, groups=groups
            ):
                train, test = labeled.iloc[train_idx], labeled.iloc[test_idx]
                model, _, X_test = _fit_tuned_model(
                    train, test, method, options, oversampling_threshold=oversampling_threshold)
                predictions.iloc[test_idx] = model.predict(X_test)
            score = evaluate_categorizer(labeled["main_category"], predictions, verbose=False)
            rows.append({"method": method, "feature_variant": variant_name,
                         "accuracy": score["accuracy"], "macro_f1": score["macro_f1"]})
    return pd.DataFrame(rows)


def tuned_within_person_cv(df, n_splits=5):
    """Nested grouped/stratified comparison with model-specific features.

    Outer folds are grouped by normalized description. Feature selection and
    parameter tuning happen independently for each model using only the outer
    training rows; the outer test rows are used once for the final score.
    """
    labeled = load_labeled(df).reset_index(drop=True)
    groups = labeled["description"].astype(str).str.strip().str.lower()
    methods = ("most_common", "knn_tfidf", "knn_embeddings",
               "random_forest", "neural_network")
    predictions = {method: pd.Series(pd.NA, index=labeled.index, dtype="string")
                   for method in methods}
    parameter_rows = []

    for fold, (train_idx, test_idx) in enumerate(
        GroupKFold(n_splits=n_splits).split(labeled, groups=groups), start=1
    ):
        train, test = labeled.iloc[train_idx], labeled.iloc[test_idx]
        predictions["most_common"].iloc[test_idx] = baseline_most_common(train, test)
        for method in methods[1:]:
            variant_name, options = _select_feature_variant(train, method)
            model, params, X_test = _fit_tuned_model(train, test, method, options)
            predictions[method].iloc[test_idx] = model.predict(X_test)
            parameter_rows.append({"fold": fold, "method": method,
                                   "feature_variant": variant_name, **params})

    scores = []
    for method, pred in predictions.items():
        result = evaluate_categorizer(labeled["main_category"], pred, verbose=False)
        scores.append({"method": method, "accuracy": result["accuracy"],
                       "macro_f1": result["macro_f1"]})
    return (pd.DataFrame(scores).set_index("method"),
            pd.DataFrame(parameter_rows), pd.DataFrame())


def write_cv_workflow_visual(path="results/media/cv_workflow.svg"):
    """Write a small, dependency-free diagram of the nested CV roles."""
    svg = """<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="810" viewBox="0 0 1120 810">
 <rect width="1120" height="810" fill="white"/>
 <style>
 text{font-family:Arial,sans-serif;font-size:16px;fill:#172033}
 .title{font-size:24px;font-weight:bold}.heading{font-size:19px;font-weight:bold}
 .small{font-size:14px}.label{font-weight:bold}.train{fill:#dcecff;stroke:#315a8a}
 .test{fill:#ffe0bd;stroke:#a85b0d}.validation{fill:#e8d9ff;stroke:#7652a1}
 .outline{fill:none;stroke:#6c7a89;stroke-width:1.5}.block{stroke-width:1.5}
 </style>
 <text x="30" y="37" class="title">How the nested cross-validation splits the data</text>
 <text x="30" y="77" class="heading">1. Split one person's labeled transaction rows into five outer folds</text>
 <text x="30" y="102" class="small">Group rows by description for splitting; models can use other selected features too.</text>
 <g text-anchor="middle" class="label">
 <rect class="train block" x="210" y="116" width="145" height="40"/><text x="282" y="142">Fold 1</text>
 <rect class="train block" x="355" y="116" width="145" height="40"/><text x="427" y="142">Fold 2</text>
 <rect class="train block" x="500" y="116" width="145" height="40"/><text x="572" y="142">Fold 3</text>
 <rect class="train block" x="645" y="116" width="145" height="40"/><text x="717" y="142">Fold 4</text>
 <rect class="train block" x="790" y="116" width="145" height="40"/><text x="862" y="142">Fold 5</text>
 </g>
 <text x="30" y="193" class="heading">2. Rotate the outer test fold</text>
 <text x="30" y="217" class="small">Each run trains and selects settings on four folds, then evaluates on the held-out fold.</text>
 <g text-anchor="middle" class="small">
 <text x="110" y="250">Outer run 1</text><text x="110" y="283">Outer run 2</text><text x="110" y="316">Outer run 3</text><text x="110" y="349">Outer run 4</text><text x="110" y="382">Outer run 5</text>
 <rect class="test block" x="210" y="230" width="145" height="27"/><rect class="train block" x="355" y="230" width="580" height="27"/><text x="282" y="249">Test 1</text><text x="645" y="249">Training folds 2, 3, 4, 5</text>
 <rect class="train block" x="210" y="263" width="145" height="27"/><rect class="test block" x="355" y="263" width="145" height="27"/><rect class="train block" x="500" y="263" width="435" height="27"/><text x="282" y="282">Train 1</text><text x="427" y="282">Test 2</text><text x="717" y="282">Train 3, 4, 5</text>
 <rect class="train block" x="210" y="296" width="290" height="27"/><rect class="test block" x="500" y="296" width="145" height="27"/><rect class="train block" x="645" y="296" width="290" height="27"/><text x="355" y="315">Train 1, 2</text><text x="572" y="315">Test 3</text><text x="790" y="315">Train 4, 5</text>
 <rect class="train block" x="210" y="329" width="435" height="27"/><rect class="test block" x="645" y="329" width="145" height="27"/><rect class="train block" x="790" y="329" width="145" height="27"/><text x="427" y="348">Train 1, 2, 3</text><text x="717" y="348">Test 4</text><text x="862" y="348">Train 5</text>
 <rect class="train block" x="210" y="362" width="580" height="27"/><rect class="test block" x="790" y="362" width="145" height="27"/><text x="500" y="381">Train 1, 2, 3, 4</text><text x="862" y="381">Test 5</text>
 </g>
 <text x="30" y="430" class="heading">3. Use inner validation to select features and tune parameters</text>
 <text x="30" y="454" class="small">Round 1: compare feature sets using fixed model settings and the inner validation rows.</text>
 <text x="30" y="478" class="small">Round 2: keep the chosen features and compare parameter settings on those same validation rows.</text>
 <g text-anchor="middle" class="small">
 <text x="110" y="512">Inner run 1</text><text x="110" y="545">Inner run 2</text><text x="110" y="578">Inner run 3</text>
 <rect class="validation block" x="210" y="492" width="242" height="27"/><rect class="train block" x="452" y="492" width="484" height="27"/><text x="331" y="511">Validation 1</text><text x="694" y="511">Train 2, 3</text>
 <rect class="train block" x="210" y="525" width="242" height="27"/><rect class="validation block" x="452" y="525" width="242" height="27"/><rect class="train block" x="694" y="525" width="242" height="27"/><text x="331" y="544">Train 1</text><text x="573" y="544">Validation 2</text><text x="815" y="544">Train 3</text>
 <rect class="train block" x="210" y="558" width="484" height="27"/><rect class="validation block" x="694" y="558" width="242" height="27"/><text x="452" y="577">Train 1, 2</text><text x="815" y="577">Validation 3</text>
 </g>
 <text x="30" y="608" class="small">Both rounds reuse these inner splits; only inner training rows are oversampled. Outer test rows stay out.</text>
 <text x="30" y="657" class="heading">4. Select, refit, and evaluate</text>
 <text x="30" y="684">Refit the selected model on the oversampled outer training data, then predict its held-out</text>
 <text x="30" y="707">outer test rows using the selected features. Repeat for each outer fold.</text>
 <text x="30" y="730">Pool all held-out predictions: accuracy is the overall correct share; macro F1 averages category F1s.</text>
 <rect class="train block" x="30" y="758" width="20" height="18"/><text x="58" y="773" class="small">Training</text>
 <rect class="validation block" x="180" y="758" width="20" height="18"/><text x="208" y="773" class="small">Inner validation</text>
 <rect class="test block" x="365" y="758" width="20" height="18"/><text x="393" y="773" class="small">Outer test</text>
 <text x="580" y="773" class="small">Fold widths are illustrative, not exact sizes.</text>
 </svg>"""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")
    return output


def cross_person_comparison(people):
    rows = []
    for train_name, test_name in (
        ("Savanna", "Jillian"),
        ("Jillian", "Savanna"),
    ):
        train = load_labeled(people[train_name])
        test = load_labeled(people[test_name])
        seen = test["description"].str.lower().isin(
            set(train["description"].str.lower())
        )

        pred = baseline_most_common(train, test)
        score = evaluate_categorizer(
            test["main_category"],
            pred,
            verbose=False,
        )
        rows.append({
            "train": train_name,
            "test": test_name,
            "method": "most_common",
            "accuracy": score["accuracy"],
            "macro_f1": score["macro_f1"],
            "familiar_descriptions": seen.mean(),
        })

        for method in ("knn_tfidf", "knn_embeddings", "random_forest", "neural_network"):
            print(f"Cross-person: train {train_name}, test {test_name}, {method}", flush=True)
            variant, options = _select_feature_variant(train, method)
            model, params, X_test = _fit_tuned_model(train, test, method, options)

            pred = model.predict(X_test)
            score = evaluate_categorizer(
                test["main_category"],
                pred,
                verbose=False,
            )
            rows.append({
                "train": train_name,
                "test": test_name,
                "method": method,
                "accuracy": score["accuracy"],
                "macro_f1": score["macro_f1"],
                "familiar_descriptions": seen.mean(),
                "feature_variant": variant,
                "status": params["status"],
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--savanna", default="savanna_transactions.csv",
                        help="Savanna labeled CSV")
    parser.add_argument("--jillian", default="jillian_transactions_labeled.csv",
                        help="Jillian labeled CSV")
    parser.add_argument("--output", default="results/tuned_comparison",
                        help="base name for CSV tables, saved in a tables subfolder")
    args = parser.parse_args()
    people = {"Savanna": load_dataset(args.savanna),
              "Jillian": load_dataset(args.jillian)}
    scores_by_person, selections = [], []
    for person, df in people.items():
        print(f"{person}: starting within-person comparison", flush=True)
        scores, params, _ = tuned_within_person_cv(df)
        scores = scores.reset_index()
        scores.insert(0, "person", person)
        params.insert(0, "person", person)
        scores_by_person.append(scores)
        selections.append(params)
        print(f"{person}: within-person comparison complete", flush=True)
    cross = cross_person_comparison(people)
    output = Path(args.output)
    table_dir = output.parent if output.parent.name == "tables" else output.parent / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "scores": pd.concat(scores_by_person, ignore_index=True),
        "selections": pd.concat(selections, ignore_index=True),
        "cross_person": cross,
    }
    for name, table in tables.items():
        path = table_dir / (output.stem + f"_{name}.csv")
        table.to_csv(path, index=False)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
