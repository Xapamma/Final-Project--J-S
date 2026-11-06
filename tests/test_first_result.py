"""Tests for the first-result baseline, KNN, and cross-person helper."""

import unittest

import pandas as pd

from ml.ml_models import (baseline_most_common, cross_person_experiment,
                          knn_tfidf, within_person_cv)


def _frame(rows):
    return pd.DataFrame(rows, columns=["description", "main_category"])


TRAIN = _frame([
    ("CHEVRON 0123 PROVO UT", "Transportation"),
    ("CHEVRON 0456 OREM UT", "Transportation"),
    ("SHELL OIL 5566 UT", "Transportation"),
    ("SMITHS FOOD 0042 UT", "Food & Dining"),
    ("SMITHS FOOD 0077 UT", "Food & Dining"),
    ("TRADER JOE S UT", "Food & Dining"),
    ("CAFE RIO PROVO", "Food & Dining"),
])


class FirstResultTest(unittest.TestCase):
    def test_most_common_predicts_training_mode(self):
        test = _frame([("ANYTHING", "Transportation")] * 3)
        self.assertEqual(baseline_most_common(TRAIN, test), ["Food & Dining"] * 3)

    def test_knn_matches_unseen_store_numbers(self):
        test = _frame([("CHEVRON 9999 SANDY UT", "?"), ("SMITHS FOOD 1234 UT", "?")])
        preds = list(knn_tfidf(TRAIN, test, n_neighbors=1))
        self.assertEqual(preds, ["Transportation", "Food & Dining"])

    def test_cross_person_reports_every_model(self):
        test = _frame([("CHEVRON 0123 PROVO UT", "Transportation"),
                       ("SMITHS FOOD 9 UT", "Food & Dining")])
        scores = cross_person_experiment(TRAIN, test)
        self.assertEqual(list(scores["model"]),
                         ["most_common", "knn_tfidf"])
        self.assertAlmostEqual(scores["familiar_descriptions"].iloc[0], 0.5)

    def test_within_person_cv_runs_knn(self):
        df = pd.concat([TRAIN] * 2, ignore_index=True)
        df["description"] = df["description"] + pd.Series(range(len(df))).astype(str).radd(" ")
        folds, overall = within_person_cv(df, lambda tr, te: knn_tfidf(tr, te, n_neighbors=1),
                                          n_splits=3)
        self.assertEqual(len(folds), 3)
        self.assertGreaterEqual(overall["accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
