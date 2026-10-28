"""Tests for the Streamlit import/export pipeline (categorization removed).

The rule-based categorization system (description maps, merchant/category
caches, General Spending fallback, auto merchants) was removed on
2026-10-27. These tests cover what remains: import, dedup, exclusions,
validation, review rows, normalizers, category vocabulary, and a
no-auto-assignment guarantee.
"""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from dashboard.bank_profiles import load_bank_profiles
from dashboard.cleaning_logic import clean_description_for_matching, smart_title
from dashboard.fuzzy_search import fuzzy_match_indices
from dashboard.transaction_import import (
    STANDARD_COLUMNS,
    build_import_signature,
    load_transactions,
    remove_source_rows,
    replace_source_rows,
)
from dashboard.final_export import (
    CATEGORY_LABEL_TO_PAIR,
    CATEGORY_PAIR_OPTIONS,
    CATEGORY_PAIR_TO_LABEL,
    CATEGORY_SUBCATEGORIES,
    FINAL_EXPORT_COLUMNS,
    FINAL_MAIN_CATEGORY_OPTIONS,
    complete_missing_categories,
    edited_row_problem,
    format_edit_problem_counts,
    format_final_export,
    get_final_export_row_issues,
    make_transfer_review_rows,
    normalize_account_type,
    normalize_bank_name,
    normalize_export_filename,
    restored_rows_to_standardized,
    split_duplicate_transactions,
    split_filtered_transactions,
    validate_final_export_rows,
)
from dashboard.dashboard_data import (
    apply_transaction_category_edits,
    analytics_transactions_csv,
    filter_transactions_by_category,
    income_for_budget_category,
    prepare_analytics_transactions,
    sort_rows_by_date,
    spending_by_main_category,
    spending_for_budget_category,
    summarize_transactions,
    trend_totals,
)


class LoadTransactionsTests(unittest.TestCase):
    """Import standardization (no categorization involved)."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.temp_path = Path(self.temp_dir.name)

    def write_csv(self, filename, content):
        path = self.temp_path / filename
        path.write_text(content, encoding="utf-8")
        return path

    def test_capital_one_split_debits_and_credits(self):
        path = self.write_csv(
            "capital_one_cc.csv",
            "Transaction Date,Posted Date,Description,Debit,Credit,Category\n"
            "2026-01-02,2026-01-03,Coffee,$12.50,,Dining\n"
            "2026-01-04,2026-01-04,Payment,,25.00,Payment\n",
        )
        result = load_transactions(path, bank="capital one")
        self.assertEqual(result.columns.tolist(), STANDARD_COLUMNS)
        self.assertEqual(result["amount"].tolist(), [-12.5, 25.0])
        self.assertEqual(result["transaction_type"].tolist(), ["debit", "credit"])
        self.assertEqual(result["account"].tolist(), ["credit card", "credit card"])

    def test_gwcu_signed_amount_and_optional_transaction_id(self):
        path = self.write_csv(
            "gwcu_checking.csv",
            '"Transaction ID","Effective Date","Posting Date","Amount",'
            '"Description","Type"\n'
            'abc-1,2026-02-01,2026-02-02,"(1,234.56)",Groceries,Card\n'
            'abc-2,2026-02-03,2026-02-03,15.25,Dividend,Dividends\n',
        )
        result = load_transactions(path, bank="gwcu")
        self.assertEqual(result["amount"].tolist(), [-1234.56, 15.25])
        self.assertEqual(result["transaction_id"].tolist(), ["abc-1", "abc-2"])

    def test_new_bank_can_be_added_with_explicit_column_mapping(self):
        path = self.write_csv(
            "new_bank.csv",
            "Date,Details,Amount\n"
            "2026-03-01,Walmart,-42.00\n"
            "2026-03-02,Payroll,1200.00\n",
        )
        result = load_transactions(
            path,
            bank="New Bank",
            account="checking",
            column_mapping={"date": "Date", "description": "Details", "amount": "Amount"},
        )
        self.assertEqual(result["description"].tolist(), ["Walmart", "Payroll"])
        self.assertEqual(result["bank"].tolist(), ["New Bank", "New Bank"])

    def test_import_signature_changes_when_file_settings_change(self):
        common = dict(
            bank="sofi",
            account="checking",
            column_mapping={"date": "Date"},
            delimiter=",",
            decimal=".",
            thousands=",",
        )
        signature_a = build_import_signature("digest", **common, positive_amounts_are_debits=False)
        signature_b = build_import_signature("digest", **common, positive_amounts_are_debits=True)
        self.assertNotEqual(signature_a, signature_b)

    def test_file_rows_can_be_removed_and_replaced_by_filename(self):
        frame = pd.DataFrame({"source_file": ["a.csv", "a.csv", "b.csv"]})
        removed = remove_source_rows(frame, source_column="source_file", source_names={"a.csv"})
        self.assertEqual(removed["source_file"].tolist(), ["b.csv"])
        replaced = replace_source_rows(
            frame,
            pd.DataFrame({"source_file": ["a.csv"]}),
            source_column="source_file",
            source_name="a.csv",
        )
        self.assertEqual(len(replaced), 2)


class BankAndSearchTests(unittest.TestCase):
    def test_bank_profiles_load_from_seed_file(self):
        profiles = load_bank_profiles()
        self.assertIsInstance(profiles, dict)
        self.assertTrue(profiles)

    def test_fuzzy_search_ignores_case_and_finds_typos(self):
        indices = fuzzy_match_indices("walmart", ["WALMART", "Taco Bell"])
        self.assertEqual(indices, [0])


class ExclusionAndDuplicateTests(unittest.TestCase):
    """Data-cleaning exclusions and dedup (not categorization)."""

    def make_standardized(self, rows):
        columns = [
            "transaction_id", "date", "posted_date", "description", "amount",
            "transaction_type", "bank_category", "status", "bank", "account",
            "currency", "source_file", "source_row",
        ]
        return pd.DataFrame(rows, columns=columns)

    def test_final_export_excludes_legacy_noise_descriptions(self):
        source = self.make_standardized(
            [
                [None, "2026-01-05", None, "Transfer to SoFi Checking", -50.0, "debit", None, None, "sofi", "savings", None, "s.csv", 1],
                [None, "2026-01-06", None, "Costco", -75.0, "debit", None, None, "sofi", "savings", None, "s.csv", 2],
            ]
        )
        exportable, filtered = split_filtered_transactions(source)
        self.assertEqual(len(exportable), 1)
        self.assertEqual(len(filtered), 1)
        self.assertIn("transfer", filtered.iloc[0]["removal_reason"].casefold())

    def test_exact_duplicate_transactions_are_excluded_for_review(self):
        source = self.make_standardized(
            [
                [None, "2026-01-05", None, "Costco", -75.0, "debit", None, None, "sofi", "savings", None, "s.csv", 1],
                [None, "2026-01-05", None, "Costco", -75.0, "debit", None, None, "sofi", "savings", None, "s.csv", 2],
            ]
        )
        included, duplicates = split_duplicate_transactions(source, pd.DataFrame())
        self.assertEqual(len(included), 1)
        self.assertEqual(len(duplicates), 1)
        self.assertIn("Duplicate", duplicates.iloc[0]["removal_reason"])

    def test_declined_status_rows_are_excluded(self):
        source = self.make_standardized(
            [
                [None, "2026-01-05", None, "Declined purchase", -75.0, "debit", None, "Declined", "sofi", "savings", None, "s.csv", 1],
            ]
        )
        included, filtered = split_filtered_transactions(source)
        self.assertTrue(included.empty)
        self.assertEqual(filtered.iloc[0]["removal_reason"], "Bank status is Declined.")


class DecategorizedExportTests(unittest.TestCase):
    """The export schema, blank-by-default categories, and vocabulary."""

    def make_transactions(self):
        return pd.DataFrame(
            [
                ["2026-01-05", "Wakanda Coffee", -4.50, "debit", "sofi", "savings"],
                ["2026-01-03", "Payroll", 1200.00, "credit", "sofi", "savings"],
            ],
            columns=["date", "description", "amount", "transaction_type", "bank", "account"],
        )

    def test_final_export_matches_schema_and_sorts_dates_ascending(self):
        exported = format_final_export(self.make_transactions())
        self.assertEqual(exported.columns.tolist(), FINAL_EXPORT_COLUMNS)
        # Dates are formatted and sorted oldest first.
        self.assertEqual(exported["date"].tolist(), ["01/03/2026", "01/05/2026"])

    def test_categories_start_blank_and_merchant_is_plain_description(self):
        exported = format_final_export(self.make_transactions())
        self.assertTrue(exported["main_category"].isna().all())
        self.assertTrue(exported["sub_category"].isna().all())
        # No merchant lookup rules: merchant is just the title-cased description.
        # (Rows are date-sorted oldest first: Payroll on 01/03 precedes coffee.)
        self.assertEqual(exported["merchant"].tolist(), ["Payroll", "Wakanda Coffee"])

    def test_rule_based_categorization_is_gone(self):
        # No General Spending fallback in the vocabulary, and no rule
        # functions in the modules the dashboard depends on.
        self.assertNotIn("General Spending", FINAL_MAIN_CATEGORY_OPTIONS)
        import dashboard.cleaning_logic as cleaning
        self.assertFalse(hasattr(cleaning, "add_categories"))
        self.assertFalse(hasattr(cleaning, "description_map"))
        self.assertFalse(hasattr(cleaning, "sub_category_map"))
        self.assertFalse(hasattr(cleaning, "category_hierarchy"))

    def test_merchant_and_category_caches_are_gone(self):
        import dashboard.final_export as export
        self.assertFalse(hasattr(export, "load_merchant_cache"))
        self.assertFalse(hasattr(export, "load_category_cache"))
        self.assertFalse(hasattr(export, "save_approved_mappings"))
        self.assertFalse(hasattr(export, "merchant_needs_review"))

    def test_transfer_stays_a_manual_choice_with_na_subcategory(self):
        df = pd.DataFrame(
            [["06/01/2026", "Transfer to savings", "Checking", "debit", 50.0, "Transfer", None, "sofi", "checking"]],
            columns=FINAL_EXPORT_COLUMNS,
        )
        completed = complete_missing_categories(df)
        self.assertEqual(completed.iloc[0]["sub_category"], "N/A")


class ValidationAndReviewTests(unittest.TestCase):
    def test_validate_accepts_complete_labeled_rows(self):
        rows = pd.DataFrame(
            [["06/01/2026", "Coffee", "Coffee Shop", "debit", -4.5, "Food & Dining", "Restaurants/Dining", "sofi", "savings"]],
            columns=FINAL_EXPORT_COLUMNS,
        )
        result, error = validate_final_export_rows(rows)
        self.assertIsNone(error)
        self.assertEqual(len(result), 1)

    def test_validate_rejects_blank_categories(self):
        rows = pd.DataFrame(
            [["06/01/2026", "Coffee", "Coffee Shop", "debit", -4.5, None, None, "sofi", "savings"]],
            columns=FINAL_EXPORT_COLUMNS,
        )
        result, error = validate_final_export_rows(rows)
        self.assertIsNotNone(error)
        self.assertIn("Main Category", error)

    def test_validate_rejects_subcategory_outside_selected_main_category(self):
        rows = pd.DataFrame(
            [["06/01/2026", "Coffee", "Coffee Shop", "debit", -4.5, "Food & Dining", "Airfare", "sofi", "savings"]],
            columns=FINAL_EXPORT_COLUMNS,
        )
        result, error = validate_final_export_rows(rows)
        self.assertIsNotNone(error)
        self.assertIn("sub-category", error)

    def test_row_issue_map_identifies_only_invalid_date_row(self):
        rows = pd.DataFrame(
            [
                ["not-a-date", "Coffee", "Coffee Shop", "debit", -4.5, "Food & Dining", "Restaurants/Dining", "sofi", "savings"],
                ["06/02/2026", "Gas", "Shell", "debit", -30.0, "Transportation", "Fuel", "sofi", "savings"],
            ],
            columns=FINAL_EXPORT_COLUMNS,
        )
        issues = get_final_export_row_issues(rows)
        self.assertEqual(list(issues), [0])
        self.assertIn("Date must use MM/DD/YYYY.", issues[0])

    def test_edited_row_problem_flags_blank_merchant_and_mismatched_category(self):
        self.assertEqual(edited_row_problem("", "Food & Dining", "Groceries"), "missing merchant")
        self.assertEqual(
            edited_row_problem("Shop", "Food & Dining", "Airfare"),
            "sub-category does not match its main category",
        )
        self.assertIsNone(edited_row_problem("Shop", "Food & Dining", "Groceries"))
        self.assertIsNone(edited_row_problem("Shop", "Transfer", "N/A"))

    def test_format_edit_problem_counts_groups_into_one_sentence(self):
        sentence = format_edit_problem_counts({"missing merchant": 2, "mismatched categories": 1})
        self.assertIn("3 row(s)", sentence)
        self.assertIn("2 missing merchant", sentence)

    def test_transfer_review_rows_build_excluded_review_row(self):
        rows = pd.DataFrame(
            [["06/01/2026", "Transfer", "Bank", "debit", 50.0, "Transfer", "N/A", "sofi", "checking"]],
            columns=FINAL_EXPORT_COLUMNS,
        )
        review = make_transfer_review_rows(
            rows,
            source_files=pd.Series(["s.csv"]),
            source_rows=pd.Series([3]),
        )
        self.assertEqual(review.iloc[0]["removal_reason"], "Manually categorized as Transfer.")

    def test_restored_dropped_row_can_be_converted_back(self):
        dropped = pd.DataFrame(
            [["06/01/2026", "Refund from shop", "Shop", "credit", 20.0, "Income", "Refunds", "sofi", "savings"]],
            columns=FINAL_EXPORT_COLUMNS,
        )
        restored, error = restored_rows_to_standardized(
            dropped,
            statuses=pd.Series(["Declined"]),
            source_files=pd.Series(["sofi.csv"]),
            source_rows=pd.Series([4]),
        )
        self.assertIsNone(error)
        self.assertEqual(restored["description"].tolist(), ["Refund from shop"])
        self.assertEqual(restored["status"].tolist(), ["Declined"])

    def test_normalize_export_filename_is_a_csv_basename(self):
        self.assertEqual(normalize_export_filename("folder/sub-name.txt"), "sub-name.csv")
        self.assertEqual(normalize_export_filename("out.csv"), "out.csv")

    def test_normalize_bank_and_account(self):
        self.assertEqual(normalize_bank_name("gwcu"), "goldenwest")
        self.assertEqual(normalize_bank_name("capital one"), "capital one")
        self.assertEqual(normalize_account_type("CC"), "credit card")
        self.assertEqual(normalize_account_type("IMM"), "money market")

    def test_label_pairs_round_trip(self):
        for main, subs in CATEGORY_SUBCATEGORIES.items():
            for sub in subs:
                pair = (main, sub)
                label = CATEGORY_PAIR_TO_LABEL[pair]
                self.assertEqual(CATEGORY_LABEL_TO_PAIR[label], pair)
                self.assertIn(label, CATEGORY_PAIR_OPTIONS)


class AnalyticsHelperTests(unittest.TestCase):
    def export_rows(self):
        return pd.DataFrame(
            [
                ["06/01/2026", "Coffee", "Coffee Shop", "debit", -4.5, "Food & Dining", "Restaurants/Dining", "sofi", "savings"],
                ["06/02/2026", "Payroll", "Payroll Deposit", "credit", 1000.0, "Income", "Paychecks", "sofi", "savings"],
                ["06/03/2026", "More food", "Mcdonalds", "debit", -8.0, "Food & Dining", "Fast Food", "sofi", "savings"],
            ],
            columns=FINAL_EXPORT_COLUMNS,
        )

    def test_prepare_analytics_transactions_handles_dates_and_amounts(self):
        prepared = prepare_analytics_transactions(self.export_rows())
        self.assertEqual(len(prepared), 3)
        totals = summarize_transactions(prepared)
        self.assertEqual(totals["income"], 1000.0)
        self.assertEqual(totals["spending"], 12.5)

    def test_spending_by_main_category_aggregates_and_orders(self):
        by_category = spending_by_main_category(prepare_analytics_transactions(self.export_rows()))
        food_row = by_category.loc[by_category["main_category"] == "Food & Dining"]
        self.assertEqual(float(food_row.iloc[0]["spending"]), 12.5)
        # Income is never listed as spending.
        self.assertNotIn("Income", by_category["main_category"].tolist())

    def test_budget_category_totals(self):
        prepared = prepare_analytics_transactions(self.export_rows())
        self.assertEqual(spending_for_budget_category(prepared, "Food & Dining"), 12.5)
        self.assertEqual(income_for_budget_category(prepared, "Income"), 1000.0)

    def test_cleaning_helper_alias_matches_new_name(self):
        from dashboard import cleaning_logic as cleaning_module
        self.assertEqual(
            clean_description_for_matching("WAL-MART #1234 UT"),
            cleaning_module.clean_description("WAL-MART #1234 UT"),
        )
        self.assertEqual(smart_title("walmart supercenter"), "Walmart Supercenter")


class AppSmokeTests(unittest.TestCase):
    """Load and render the process-statement app page once, with no uploads."""

    def test_app_page_runs_without_exceptions_when_empty(self):
        from streamlit.testing.v1 import AppTest

        app_path = Path(__file__).resolve().parent.parent / "app.py"
        app = AppTest.from_file(str(app_path))
        app.run(timeout=30)
        self.assertFalse(app.exception)