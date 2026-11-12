"""Working investigation: biggest confusion pairs, TF-IDF vs embedding KNN.

Run from the project root (labeled CSVs are private):
    uv run python -m ml.confusion_analysis --jillian jillian_transactions_labeled.csv \
                                           --savanna savanna_transactions.csv

Two GroupKFold groupings are compared:
  * exact      - group by the exact description (what first_result.py uses).
  * normalized - group by the description with digits and punctuation removed,
                 so "VENMO ... ID:1040116242842" and "VENMO ... ID:1041142349380",
                 or "KEEP THE CHANGE ... FOR 02/03/25" and "... FOR 02/18/25",
                 are the same group and never sit on both sides of a split.

Writes results/confusion_analysis.md (counts and short merchant keys only).
Row-level detail goes to .data/confusion_detail.csv, which is gitignored.
"""

import argparse
import os
import re

import pandas as pd

from ml.ml_models import (FIRST_RESULT_MODELS, evaluate_categorizer, load_dataset,
                          load_labeled, within_person_cv)

TARGET = "main_category"
OUTPUT = os.path.join("results", "confusion_analysis.md")
DETAIL = os.path.join(".data", "confusion_detail.csv")
MODELS = ["most_common", "knn_tfidf", "knn_embeddings"]


def group_key(description):
    """Lowercase, drop digits/punctuation, collapse spaces."""
    text = re.sub(r"[^a-z]+", " ", str(description).lower())
    return re.sub(r"\s+", " ", text).strip()


_PREFIX = re.compile(r"^(checkcard|mobile purchase|purchase|tst|sq|py|cpi|tm|sp)\b\s*", re.I)


# Merchants or places that would identify where we live, study, or work out.
_REDACT = [(re.compile(r"\b(BYU|BRIGHAM YOUNG)\b.*", re.I), "UNIVERSITY"),
           (re.compile(r"\bEOS FITNESS\b.*", re.I), "GYM"),
           (re.compile(r"\b(PROVO|OREM)\b", re.I), "")]


def merchant_key(description):
    """Short, non-identifying merchant label for the committed report."""
    text = re.split(r"\b(INDN|ID|DES|CO ID|Confirmation)\b", str(description))[0]
    text = re.sub(r"[^A-Za-z&' ]+", " ", text).strip()
    for _ in range(2):
        text = _PREFIX.sub("", text).strip()
    for pattern, replacement in _REDACT:
        text = pattern.sub(replacement, text).strip()
    return " ".join(text.split()[:3]).upper() or "(blank)"


def oof_predictions(labeled, fit_predict, group_col):
    """Same split as within_person_cv, but keep each row's prediction."""
    from sklearn.model_selection import GroupKFold

    groups = labeled[group_col].astype(str).str.strip().str.lower()
    out = pd.Series(pd.NA, index=labeled.index, dtype="string")
    for train_idx, test_idx in GroupKFold(n_splits=5).split(labeled, groups=groups):
        train, test = labeled.iloc[train_idx], labeled.iloc[test_idx]
        out.loc[test.index] = list(fit_predict(train, test))
    return out


def person_section(name, df):
    labeled = load_labeled(df).reset_index(drop=True)
    labeled["group_key"] = labeled["description"].map(group_key)
    lines = [f"## {name}", "",
             f"{len(labeled)} labeled rows; {labeled['description'].str.lower().nunique()} exact "
             f"descriptions vs {labeled['group_key'].nunique()} normalized descriptions.", ""]

    # 1. Scores under both groupings.
    rows = []
    for grouping, col in [("exact", "description"), ("normalized", "group_key")]:
        for model in MODELS:
            folds, _ = within_person_cv(labeled, FIRST_RESULT_MODELS[model],
                                        target=TARGET, group_col=col)
            rows.append({"grouping": grouping, "model": model,
                         "accuracy": folds["accuracy"].mean(),
                         "macro_f1": folds["macro_f1"].mean()})
    scores = pd.DataFrame(rows).pivot(index="model", columns="grouping",
                                      values=["accuracy", "macro_f1"]).round(3)
    scores = scores.loc[MODELS]
    scores.columns = [f"{m}_{g}" for m, g in scores.columns]
    lines += ["Mean of 5 GroupKFold folds:", "", "```", scores.to_string(), "```", ""]

    # 2. Row-level predictions under the stricter (normalized) grouping.
    for model in ["knn_tfidf", "knn_embeddings"]:
        labeled[f"pred_{model}"] = oof_predictions(labeled, FIRST_RESULT_MODELS[model], "group_key")
    labeled["merchant"] = labeled["description"].map(merchant_key)

    # 3. Per-category F1, TF-IDF vs embeddings.
    f1 = pd.DataFrame({
        "support": labeled[TARGET].value_counts(),
        "f1_tfidf": evaluate_categorizer(labeled[TARGET], labeled["pred_knn_tfidf"],
                                         verbose=False)["report"]["f1-score"],
        "f1_embeddings": evaluate_categorizer(labeled[TARGET], labeled["pred_knn_embeddings"],
                                              verbose=False)["report"]["f1-score"],
    }).dropna(subset=["support"]).sort_values("support", ascending=False)
    f1["better"] = f1.apply(lambda r: "embeddings" if r.f1_embeddings > r.f1_tfidf + 0.02
                            else ("tfidf" if r.f1_tfidf > r.f1_embeddings + 0.02 else "tie"), axis=1)
    f1["support"] = f1["support"].astype(int)
    lines += ["Per-category F1 (normalized grouping, pooled out-of-fold):", "", "```",
              f1.round(3).to_string(), "```", ""]

    # 4. Biggest confusion pairs.
    pair_counts = {}
    for model in ["knn_tfidf", "knn_embeddings"]:
        wrong = labeled[labeled[TARGET] != labeled[f"pred_{model}"]]
        pair_counts[model] = wrong.groupby([TARGET, f"pred_{model}"]).size()
    pairs = pd.DataFrame(pair_counts).fillna(0).astype(int)
    pairs.index.names = ["true", "predicted"]
    pairs["total"] = pairs.sum(axis=1)
    top = pairs.sort_values("total", ascending=False).head(8)
    lines += ["Biggest confusion pairs (row counts, normalized grouping):", "",
              "| true | predicted | TF-IDF | embeddings | top merchants (embeddings errors) |",
              "|---|---|---:|---:|---|"]
    for (true, pred), r in top.iterrows():
        rows_ = labeled[(labeled[TARGET] == true) & (labeled["pred_knn_embeddings"] == pred)]
        merchants = rows_["merchant"].value_counts().head(4)
        m_text = ", ".join(f"{m} ({n})" for m, n in merchants.items()) or "-"
        lines.append(f"| {true} | {pred} | {r.knn_tfidf} | {r.knn_embeddings} | {m_text} |")
    lines.append("")

    labeled["person"] = name
    detail = labeled[["person", "date", "description", "amount", TARGET,
                      "pred_knn_tfidf", "pred_knn_embeddings", "merchant", "group_key"]]
    return lines, detail


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--jillian")
    parser.add_argument("--savanna")
    args = parser.parse_args()

    lines = ["# Working investigation: confusion pairs and TF-IDF vs embeddings", "",
             "Rerun: `uv run python -m ml.confusion_analysis --jillian <csv> --savanna <csv>`. "
             "`exact` groups GroupKFold folds by the exact description (as in first_result.md); "
             "`normalized` groups by the description with digits and punctuation removed, so "
             "rows that differ only by an ID, date, or confirmation number stay in the same fold.", ""]
    details = []
    for name, path in [("Jillian", args.jillian), ("Savanna", args.savanna)]:
        if not path:
            continue
        section, detail = person_section(name, load_dataset(path))
        lines += section
        details.append(detail)
    os.makedirs("results", exist_ok=True)
    with open(OUTPUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.makedirs(".data", exist_ok=True)
    pd.concat(details).to_csv(DETAIL, index=False)
    print("\n".join(lines))
    print(f"\nSaved -> {OUTPUT} (row detail, private -> {DETAIL})")


if __name__ == "__main__":
    main()
