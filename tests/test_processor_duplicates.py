"""Tests for the processor's flag-then-approve duplicate review."""

import os
import tempfile
import unittest

import pandas as pd

from ml.bank_statment_processor import DUPLICATE_REVIEW_FILE, review_duplicates


def _transactions():
    rows = [
        # Two identical charges (same file) - could be real repeat purchases.
        ("2025-07-12", "PHOTO BOOTH", -5.0, "a.csv"),
        ("2025-07-12", "PHOTO BOOTH", -5.0, "a.csv"),
        # Same transaction in two overlapping exports.
        ("2025-05-01", "GROCERY", -20.0, "jan_jun.csv"),
        ("2025-05-01", "GROCERY", -20.0, "may_dec.csv"),
        # Not a duplicate.
        ("2025-05-02", "GAS", -30.0, "may_dec.csv"),
    ]
    df = pd.DataFrame(rows, columns=["date", "raw_description", "amount", "source_file"])
    df["date"] = pd.to_datetime(df["date"])
    df["bank"] = "bank"
    df["account"] = "credit card"
    return df


class ReviewDuplicatesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = self.tmp.name
        self.review_path = os.path.join(self.folder, DUPLICATE_REVIEW_FILE)

    def tearDown(self):
        self.tmp.cleanup()

    def _set_decision(self, description, decision):
        review = pd.read_csv(self.review_path, keep_default_na=False, dtype=str)
        review.loc[review["raw_description"] == description, "decision"] = decision
        review.to_csv(self.review_path, index=False)

    def test_flags_but_keeps_everything_until_approved(self):
        result = review_duplicates(_transactions(), self.folder)
        self.assertEqual(len(result), 5)
        review = pd.read_csv(self.review_path, keep_default_na=False)
        self.assertEqual(len(review), 2)
        self.assertTrue((review["decision"] == "").all())
        found = dict(zip(review["raw_description"], review["found_in"]))
        self.assertEqual(found, {"PHOTO BOOTH": "same file", "GROCERY": "different files"})

    def test_remove_drops_extra_copies_and_keep_keeps_them(self):
        review_duplicates(_transactions(), self.folder)
        self._set_decision("GROCERY", "remove")
        self._set_decision("PHOTO BOOTH", "keep")

        result = review_duplicates(_transactions(), self.folder)
        self.assertEqual(len(result), 4)
        self.assertEqual((result["raw_description"] == "GROCERY").sum(), 1)
        self.assertEqual((result["raw_description"] == "PHOTO BOOTH").sum(), 2)

        # Decisions survive a rerun.
        review = pd.read_csv(self.review_path, keep_default_na=False)
        self.assertEqual(
            dict(zip(review["raw_description"], review["decision"])),
            {"GROCERY": "remove", "PHOTO BOOTH": "keep"},
        )

    def test_rejects_unknown_decision(self):
        review_duplicates(_transactions(), self.folder)
        self._set_decision("GROCERY", "yes")
        with self.assertRaises(ValueError):
            review_duplicates(_transactions(), self.folder)


if __name__ == "__main__":
    unittest.main()
