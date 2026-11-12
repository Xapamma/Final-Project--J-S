"""
First runnable result for the data checkpoint: a most-common baseline plus
sentence-embedding KNN and a TF-IDF KNN comparison, predicting main_category
from the transaction description and amount. Future ML models will use the
sentence-embedding features.

    1. Within each person: GroupKFold (5 folds) grouped by description, so
       the same description text is never in both train and test.
    2. Cross-person (if both files are given): train on one, test on the
       other, in both directions.

Writes results/first_result.md with scores and confusion matrices. Only
category names and counts are written - no descriptions or amounts.

Run from the project root (point each flag at that person's LABELED csv):
    uv run python -m ml.first_result --savanna savanna_transactions.csv
    uv run python -m ml.first_result --jillian jillian_transactions_labeled.csv \
                                     --savanna savanna_transactions.csv
"""

import argparse
import os

import pandas as pd

from ml.ml_models import (FIRST_RESULT_MODELS, cross_person_experiment,
                          evaluate_categorizer, load_dataset, load_labeled,
                          within_person_cv)

TARGET = "main_category"
OUTPUT = os.path.join("results", "first_result.md")


def _table(df):
    return "```\n" + df.to_string() + "\n```"


# Short column headers so the confusion matrix fits on screen.
ABBREVIATIONS = {
    "Education": "Edu", "Entertainment": "Ent", "Food & Dining": "Food",
    "Gifts & Donations": "Gift", "Health & Wellness": "Hlth",
    "Housing & Bills": "Hous", "Income": "Inc", "Miscellaneous": "Misc",
    "Savings & Investments": "Sav", "Shopping & Supplies": "Shop",
    "Transfer": "Xfer", "Transportation": "Trans", "Travel & Lodging": "Trvl",
}


def _abbrev(category):
    return ABBREVIATIONS.get(category) or "".join(w[0] for w in category.split() if w[0].isalnum())


def _confusion_table(cm):
    """Markdown confusion matrix: short column codes, '·' for zeros, bold diagonal."""
    codes = [_abbrev(c) for c in cm.columns]
    lines = ["| true ↓ / predicted → | " + " | ".join(codes) + " | total |",
             "|---|" + "---:|" * (len(codes) + 1)]
    for true in cm.index:
        cells = []
        for pred in cm.columns:
            n = int(cm.loc[true, pred])
            cell = "·" if n == 0 else str(n)
            cells.append(f"**{cell}**" if true == pred else cell)
        lines.append(f"| {true} ({_abbrev(true)}) | " + " | ".join(cells)
                     + f" | {int(cm.loc[true].sum())} |")
    return "\n".join(lines)


def within_person_section(name, df):
    labeled = load_labeled(df)
    lines = [f"### {name}: within-person (GroupKFold, 5 folds, grouped by description)", "",
             f"{len(labeled)} labeled rows, {labeled['description'].str.lower().nunique()} distinct descriptions, "
             f"{labeled[TARGET].nunique()} main categories.", ""]
    rows, embedding_overall = [], None
    for model_name, fit_predict in FIRST_RESULT_MODELS.items():
        folds, overall = within_person_cv(df, fit_predict, target=TARGET)
        rows.append({
            "model": model_name,
            "accuracy_mean": folds["accuracy"].mean(),
            "accuracy_sd": folds["accuracy"].std(),
            "macro_f1_mean": folds["macro_f1"].mean(),
            "macro_f1_sd": folds["macro_f1"].std(),
        })
        if model_name == "knn_embeddings":
            embedding_overall = overall
    scores = pd.DataFrame(rows).set_index("model").round(3)
    lines += [_table(scores), "",
              "Embedding KNN per-category report (pooled out-of-fold predictions):", "",
              _table(embedding_overall["report"].round(3)), "",
              "Embedding KNN confusion matrix (rows = true category, columns = predicted; "
              "bold diagonal = correct, · = 0):", "",
              _confusion_table(embedding_overall["confusion_matrix"]), ""]
    print(scores.to_string())
    return lines


def cross_person_section(train_name, train_df, test_name, test_df):
    scores = cross_person_experiment(train_df, test_df, target=TARGET)
    familiar = scores["familiar_descriptions"].iloc[0]
    table = scores.drop(columns="familiar_descriptions").set_index("model").round(3)
    print(f"\ntrain {train_name} -> test {test_name}\n{table.to_string()}")
    return [f"### Cross-person: train {train_name} -> test {test_name}", "",
            f"{familiar:.1%} of {test_name}'s rows have a description that also appears in "
            f"{train_name}'s data.", "", _table(table), ""]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jillian", help="Jillian's labeled CSV")
    parser.add_argument("--savanna", help="Savanna's labeled CSV")
    parser.add_argument("--output", default=OUTPUT,
                        help=f"where to write the markdown report (default {OUTPUT})")
    args = parser.parse_args()
    people = {name: load_dataset(path) for name, path in
              [("Jillian", args.jillian), ("Savanna", args.savanna)] if path}
    if not people:
        parser.error("give at least one of --jillian / --savanna")

    lines = ["# First result: baseline, TF-IDF KNN, and embedding KNN (main_category)", "",
             "Models: `most_common` (always the most frequent training label), "
             "`knn_tfidf` (character TF-IDF + standardized amount multiplied by 25 "
             "+ 5-NN, cosine), and "
             "`knn_embeddings` (all-MiniLM-L6-v2 sentence embeddings + standardized "
             "amount multiplied by 25 + 5-NN, cosine). "
             "Rerun: `uv run python -m ml.first_result --jillian <csv> --savanna <csv>`", ""]
    for name, df in people.items():
        print(f"\n== {name} ==")
        lines += within_person_section(name, df)

    if len(people) == 2:
        (a, df_a), (b, df_b) = people.items()
        lines += cross_person_section(b, df_b, a, df_a)
        lines += cross_person_section(a, df_a, b, df_b)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
