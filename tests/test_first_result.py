"""Tests for the first-result baseline, KNN, and cross-person helper."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ml.ml_models import (baseline_most_common, cross_person_experiment,
                          knn_embeddings, knn_tfidf, within_person_cv)


def _frame(rows):
    return pd.DataFrame(rows, columns=["description", "main_category", "amount"])


class _FakeEmbeddingModel:
    def encode(self, descriptions, **_kwargs):
        return _fake_embedding_values(descriptions)


def _fake_embedding_values(descriptions):
    return np.array([
        [1.0, 0.0] if any(token in str(description) for token in ("CHEVRON", "SHELL"))
        else [0.0, 1.0]
        for description in descriptions
    ])


def _fake_embeddings(descriptions, _model_name):
    return _fake_embedding_values(descriptions), _FakeEmbeddingModel()


TRAIN = _frame([
    ("CHEVRON 0123 PROVO UT", "Transportation", -40.0),
    ("CHEVRON 0456 OREM UT", "Transportation", -35.0),
    ("SHELL OIL 5566 UT", "Transportation", -45.0),
    ("SMITHS FOOD 0042 UT", "Food & Dining", -80.0),
    ("SMITHS FOOD 0077 UT", "Food & Dining", -75.0),
    ("TRADER JOE S UT", "Food & Dining", -60.0),
    ("CAFE RIO PROVO", "Food & Dining", -25.0),
])


class FirstResultTest(unittest.TestCase):
    def test_most_common_predicts_training_mode(self):
        test = _frame([("ANYTHING", "Transportation", -10.0)] * 3)
        self.assertEqual(baseline_most_common(TRAIN, test), ["Food & Dining"] * 3)

    def test_knn_matches_unseen_store_numbers(self):
        test = _frame([
            ("CHEVRON 9999 SANDY UT", "?", -42.0),
            ("SMITHS FOOD 1234 UT", "?", -78.0),
        ])
        with patch("ml.ml_models.embed_descriptions", side_effect=_fake_embeddings):
            preds = list(knn_embeddings(TRAIN, test, n_neighbors=1))
        self.assertEqual(preds, ["Transportation", "Food & Dining"])

    def test_tfidf_matches_unseen_store_numbers(self):
        test = _frame([
            ("CHEVRON 9999 SANDY UT", "?", -42.0),
            ("SMITHS FOOD 1234 UT", "?", -78.0),
        ])
        preds = list(knn_tfidf(TRAIN, test, n_neighbors=1, amount_weight=0))
        self.assertEqual(preds, ["Transportation", "Food & Dining"])

    def test_cross_person_reports_every_model(self):
        test = _frame([
            ("CHEVRON 0123 PROVO UT", "Transportation", -40.0),
            ("SMITHS FOOD 9 UT", "Food & Dining", -70.0),
        ])
        with patch("ml.ml_models.embed_descriptions", side_effect=_fake_embeddings):
            scores = cross_person_experiment(TRAIN, test)
        self.assertEqual(list(scores["model"]),
                         ["most_common", "knn_tfidf", "knn_embeddings"])
        self.assertAlmostEqual(scores["familiar_descriptions"].iloc[0], 0.5)

    def test_within_person_cv_runs_knn(self):
        df = pd.concat([TRAIN] * 2, ignore_index=True)
        df["description"] = df["description"] + pd.Series(range(len(df))).astype(str).radd(" ")
        with patch("ml.ml_models.embed_descriptions", side_effect=_fake_embeddings):
            folds, overall = within_person_cv(
                df,
                lambda tr, te: knn_embeddings(tr, te, n_neighbors=1),
                n_splits=3,
            )
        self.assertEqual(len(folds), 3)
        self.assertGreaterEqual(overall["accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
