"""Compare fixed feature sets on paired splits without saving transactions.

Run: python -m ml.feature_ablation
The output contains only aggregate scores and model configuration names.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import ml.ml_models as ml
from ml.model_comparison import _build_features, _default_model_params, _model_method

METHODS = ("knn_tfidf", "knn_embeddings", "random_forest", "neural_network")


def variants(method):
    if method == "knn_tfidf":
        return [("description", {"amount_weight": 0.0}),
                ("description + amount", {"amount_weight": 1.0})]
    base = {"include_amount": False, "include_type": False,
            "include_account": False, "amount_weight": 1.0}
    return [("description", dict(base)),
            ("description + amount", {**base, "include_amount": True}),
            ("description + type", {**base, "include_type": True}),
            ("description + account", {**base, "include_account": True})]


def _predict(train, test, method, options):
    X_train, X_test = _build_features(method, train, test, options)
    X_train, y_train = ml.oversample_small_categories(
        X_train, train["main_category"], minimum_support=20, random_state=42)
    model = ml.train_categorizer(X_train, y_train, method=_model_method(method),
                                 **_default_model_params(method))
    return model.predict(X_test)


def _cache_embeddings(people):
    """Cache outputs of the fixed pretrained encoder, never labels or fitted features."""
    descriptions = pd.concat([df["description"] for df in people.values()])
    values = descriptions.fillna("").astype(str).unique().tolist()
    encoded, _ = ml.embed_descriptions(values)
    lookup = dict(zip(values, encoded))

    class Encoder:
        def encode(self, descriptions, **kwargs):
            return np.stack([lookup[str(value)] for value in descriptions])

    encoder = Encoder()
    def cached(descriptions, model_name=ml.DEFAULT_EMBEDDING_MODEL):
        values = pd.Series(descriptions).fillna("").astype(str).tolist()
        return encoder.encode(values), encoder
    ml.embed_descriptions = cached


def run(people, output, *, resume=False):
    """Write aggregate scores after every completed feature-set comparison."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if output.exists() and not resume:
        raise FileExistsError(f"{output} already exists; use --resume only with the same data and settings, or choose a new --output.")
    if output.exists():
        rows = pd.read_csv(output).to_dict("records")
    completed = {(r["setting"], r["train_person"], r["test_person"],
                  r["method"], r["feature_variant"]) for r in rows if r["fold"] == "pooled"}

    def save(new_rows):
        rows.extend(new_rows)
        pd.DataFrame(rows).to_csv(output, index=False)

    for train_name, train_data in people.items():
        train_data = ml.load_labeled(train_data).reset_index(drop=True)
        groups = train_data["description"].str.strip().str.lower()
        splits = list(GroupKFold(n_splits=5).split(train_data, groups=groups))
        other_name = next(name for name in people if name != train_name)
        other = ml.load_labeled(people[other_name]).reset_index(drop=True)
        for setting, test_name in (("within", train_name), ("cross", other_name)):
            key = (setting, train_name, test_name, "most_common", "none")
            if key in completed:
                continue
            base = {"train_person": train_name, "test_person": test_name,
                    "method": "most_common", "feature_variant": "none", "setting": setting}
            if setting == "within":
                predictions = pd.Series(index=train_data.index, dtype="string")
                result_rows = []
                for fold, (tr, te) in enumerate(splits, 1):
                    prediction = ml.baseline_most_common(train_data.iloc[tr], train_data.iloc[te])
                    predictions.iloc[te] = prediction
                    score = ml.evaluate_categorizer(train_data.iloc[te]["main_category"], prediction, verbose=False)
                    result_rows.append({**base, "fold": str(fold), "n_test": len(te),
                                        "accuracy": score["accuracy"], "macro_f1": score["macro_f1"]})
                truth = train_data["main_category"]
            else:
                predictions = ml.baseline_most_common(train_data, other)
                truth = other["main_category"]
                result_rows = []
            score = ml.evaluate_categorizer(truth, predictions, verbose=False)
            result_rows.append({**base, "fold": "pooled", "n_test": len(truth),
                                "accuracy": score["accuracy"], "macro_f1": score["macro_f1"]})
            save(result_rows)
        for method in METHODS:
            for feature_name, options in variants(method):
                base = {"train_person": train_name, "method": method,
                        "feature_variant": feature_name}
                key = ("within", train_name, train_name, method, feature_name)
                if key not in completed:
                    print(f"Within {train_name}: {method}, {feature_name}", flush=True)
                    predictions = pd.Series(index=train_data.index, dtype="string")
                    result_rows = []
                    for fold, (tr, te) in enumerate(splits, 1):
                        prediction = _predict(train_data.iloc[tr], train_data.iloc[te], method, options)
                        predictions.iloc[te] = prediction
                        score = ml.evaluate_categorizer(train_data.iloc[te]["main_category"], prediction, verbose=False)
                        result_rows.append({**base, "setting": "within", "test_person": train_name,
                                            "fold": str(fold), "n_test": len(te),
                                            "accuracy": score["accuracy"], "macro_f1": score["macro_f1"]})
                    score = ml.evaluate_categorizer(train_data["main_category"], predictions, verbose=False)
                    result_rows.append({**base, "setting": "within", "test_person": train_name,
                                        "fold": "pooled", "n_test": len(train_data),
                                        "accuracy": score["accuracy"], "macro_f1": score["macro_f1"]})
                    save(result_rows)
                key = ("cross", train_name, other_name, method, feature_name)
                if key not in completed:
                    print(f"Cross {train_name} -> {other_name}: {method}, {feature_name}", flush=True)
                    prediction = _predict(train_data, other, method, options)
                    score = ml.evaluate_categorizer(other["main_category"], prediction, verbose=False)
                    save([{**base, "setting": "cross", "test_person": other_name,
                           "fold": "pooled", "n_test": len(other),
                           "accuracy": score["accuracy"], "macro_f1": score["macro_f1"]}])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--savanna", default="savanna_transactions.csv")
    parser.add_argument("--jillian", default="jillian_transactions_labeled.csv")
    parser.add_argument("--output", default="results/tables/feature_ablation_scores.csv")
    parser.add_argument("--resume", action="store_true",
                        help="resume an interrupted run only if data and settings are unchanged")
    args = parser.parse_args()
    people = {"Savanna": ml.load_dataset(args.savanna), "Jillian": ml.load_dataset(args.jillian)}
    _cache_embeddings(people)
    run(people, args.output, resume=args.resume)


if __name__ == "__main__":
    main()
