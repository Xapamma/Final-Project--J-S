"""Tests for the first-result baseline, KNN, and cross-person helper."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ml.ml_models import (AMOUNT_WEIGHT_GRID, baseline_most_common, cross_person_experiment,
                          CV_PARAM_GRIDS, knn_embeddings, knn_tfidf, train_categorizer,
                          oversample_small_categories, build_embedding_feature_ablation,
                          tune_amount_multiplier, tune_knn, within_person_cv)


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
    def test_final_amount_search_uses_only_zero_and_one(self):
        self.assertEqual(AMOUNT_WEIGHT_GRID, [0.0, 1.0])

    def test_oversampling_raises_small_categories_to_threshold(self):
        X = np.array([[0.0], [1.0], [2.0]])
        y = pd.Series(["A", "A", "B"])
        X_resampled, y_resampled = oversample_small_categories(
            X, y, minimum_support=4, random_state=42
        )
        self.assertEqual(len(X_resampled), 8)
        self.assertEqual(y_resampled.value_counts()["A"], 4)
        self.assertEqual(y_resampled.value_counts()["B"], 4)

    def test_feature_ablation_adds_only_requested_features(self):
        train = pd.concat([_frame([
            ("CHEVRON 1", "Transportation", -40.0),
            ("SMITHS 1", "Food & Dining", -70.0),
        ])], ignore_index=True)
        test = _frame([
            ("CHEVRON 2", "?", -42.0),
        ])
        train["type"] = ["debit", "debit"]
        train["account"] = ["checking", "credit card"]
        test["type"] = ["credit"]
        test["account"] = ["checking"]
        with patch("ml.ml_models.embed_descriptions", side_effect=_fake_embeddings):
            base, base_test, _ = build_embedding_feature_ablation(
                train, test
            )
            with_type, with_type_test, _ = build_embedding_feature_ablation(
                train, test, include_type=True
            )
        self.assertEqual(base.shape[1], 2)
        self.assertGreater(with_type.shape[1], base.shape[1])
        self.assertEqual(base_test.shape[0], 1)
        self.assertEqual(with_type_test.shape[0], 1)

    def test_neural_network_grid_includes_regularization_and_early_stopping(self):
        grid = CV_PARAM_GRIDS["neural_network"]
        self.assertIn("alpha", grid)
        self.assertIn("learning_rate_init", grid)
        self.assertIn("early_stopping", grid)
        self.assertIn(True, grid["early_stopping"])
        self.assertIn(1000, grid["max_iter"])

    def test_train_categorizer_supports_all_models(self):
        X = np.array([[0.0], [0.1], [1.0], [1.1]])
        y = np.array(["A", "A", "B", "B"])
        knn = train_categorizer(X, y, method="knn", n_neighbors=1)
        forest = train_categorizer(X, y, method="random_forest", n_estimators=10)
        network = train_categorizer(
            X,
            y,
            method="neural_network",
            hidden_layer_sizes=(5,),
            max_iter=100,
            early_stopping=False,
        )
        self.assertEqual(len(knn.predict(X)), 4)
        self.assertEqual(len(forest.predict(X)), 4)
        self.assertEqual(len(network.predict(X)), 4)

    def test_knn_parameter_tuning_finds_cv_parameters(self):
        X = np.array([[0.0], [0.1], [0.9], [1.0], [1.1], [1.9]])
        y = np.array(["A", "A", "B", "B", "C", "C"])
        search = tune_knn(
            X,
            y,
            param_grid={"n_neighbors": [1, 3], "metric": ["euclidean"]},
            cv=2,
        )
        self.assertIn("n_neighbors", search.best_params_)
        self.assertTrue(hasattr(search.best_estimator_, "predict"))

    def test_amount_multiplier_and_knn_parameters_are_tuned(self):
        train = pd.concat([
            _frame([
                ("CHEVRON 1", "Transportation", -40.0),
                ("CHEVRON 2", "Transportation", -42.0),
                ("SMITHS 1", "Food & Dining", -70.0),
                ("SMITHS 2", "Food & Dining", -72.0),
                ("CAFE 1", "Food & Dining", -20.0),
                ("CAFE 2", "Food & Dining", -22.0),
            ])
        ], ignore_index=True)
        with patch("ml.ml_models.embed_descriptions", side_effect=_fake_embeddings):
            best, cv_results = tune_amount_multiplier(
                train,
                feature_type="embedding",
                amount_weights=[0.0, 25.0],
                knn_param_grid={
                    "n_neighbors": [1, 3],
                    "metric": ["cosine"],
                },
                cv=2,
            )
        self.assertIn(best["amount_weight"], [0.0, 25.0])
        self.assertIn(best["n_neighbors"], [1, 3])
        self.assertEqual(len(cv_results), 4)

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
