import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from ml.model_comparison import _fit_tuned_model, main


class TuningIsolationTest(unittest.TestCase):
    def test_comparison_writes_three_data_tables_without_markdown(self):
        scores = pd.DataFrame({"accuracy": [0.5], "macro_f1": [0.4]},
                              index=pd.Index(["model"], name="method"))
        selections = pd.DataFrame({"fold": [1], "method": ["model"]})
        cross = pd.DataFrame({"train": ["Savanna"], "test": ["Jillian"]})
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "comparison.md"
            with patch("sys.argv", ["comparison", "--output", str(base)]), \
                 patch("ml.model_comparison.load_dataset", return_value=pd.DataFrame()), \
                 patch("ml.model_comparison.tuned_within_person_cv", side_effect=lambda df: (scores.copy(), selections.copy(), None)), \
                 patch("ml.model_comparison.cross_person_comparison", return_value=cross):
                main()
            self.assertFalse(base.exists())
            for suffix in ("scores", "selections", "cross_person"):
                self.assertTrue((Path(directory) / "tables" / f"comparison_{suffix}.csv").exists())

    def test_preprocessing_and_oversampling_use_training_rows_only(self):
        train = pd.DataFrame({"description": list("abcdef"), "amount": range(6),
                              "main_category": ["A", "A", "A", "B", "B", "B"]})
        test = pd.DataFrame({"description": ["outside"], "amount": [7],
                             "main_category": ["A"]})
        prepared, oversampled = [], []

        def features(method, tr, val, options):
            prepared.append((set(tr.description), set(val.description)))
            return np.array(tr.description).reshape(-1, 1), np.array(val.description).reshape(-1, 1)

        def oversample(x, y, **kwargs):
            oversampled.append(set(x[:, 0]))
            return x, y

        class Model:
            def predict(self, x):
                return np.array(["A"] * len(x))

        grid = {"knn": {"n_neighbors": [1], "metric": ["cosine"]}}
        with patch("ml.model_comparison._build_features", side_effect=features), \
             patch("ml.model_comparison.oversample_small_categories", side_effect=oversample), \
             patch("ml.model_comparison.train_categorizer", return_value=Model()), \
             patch("ml.model_comparison.CV_PARAM_GRIDS", grid):
            _, params, _ = _fit_tuned_model(train, test, "knn_tfidf", {"amount_weight": 0})

        self.assertEqual(params["status"], "tuned")
        self.assertEqual(len(prepared), 4)
        for (tr, validation), sampled in zip(prepared, oversampled):
            self.assertTrue(tr.isdisjoint(validation))
            self.assertEqual(sampled, tr)
            self.assertNotIn("outside", sampled)
        self.assertEqual(prepared[-1][1], {"outside"})


if __name__ == "__main__":
    unittest.main()
