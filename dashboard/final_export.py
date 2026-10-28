"""Convert standardized transactions to the project's categorized CSV format.

DE-CATEGORIZED (2026-10-27)
---------------------------
The rule-based categorization system is gone:
  * no description_map / sub_category_map / category_hierarchy rules,
  * no merchant_cache.json / category_cache.json lookups,
  * no automatic merchant identification,
  * no "General Spending" / "Other Income" fallback defaults.

`main_category` / `sub_category` start BLANK on purpose. Users pick their own
category pairs in the Streamlit UI (that labeling step is what creates the
training data for `ml_models.py`), and `merchant` is just the cleaned
description unless a user edits it.

What remains: the export schema, the category *vocabulary* used for the UI
dropdowns (never auto-assigned), the transfer/declined/noise exclusions, the
duplicate detection, row validation, and the dropped-transaction review rows.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

import pandas as pd

from dashboard.cleaning_logic import clean_description_for_matching, smart_title
from dashboard.transaction_import import STANDARD_COLUMNS


FINAL_EXPORT_COLUMNS = [
    "date",
    "description",
    "merchant",
    "type",
    "amount",
    "main_category",
    "sub_category",
    "bank",
    "account",
]

# Category vocabulary used ONLY as selectable options in the UI / for
# budget planning. Nothing is ever assigned automatically from this dict.
CATEGORY_SUBCATEGORIES = {
    "Transfer": ["N/A"],
    "Food & Dining": ["Groceries", "Restaurants/Dining", "Fast Food", "Snacks"],
    "Transportation": [
        "Vehicle Payments",
        "Auto Insurance",
        "Fuel",
        "Licensing",
        "Parking",
        "Repairs",
        "Maintenance",
    ],
    "Health & Wellness": [
        "Doctor/Dentist",
        "Medicine/Drugs",
        "Personal Care",
        "Health Insurance",
    ],
    "Travel & Lodging": ["Airfare", "Lodging", "Travel & Commute"],
    "Shopping & Supplies": [
        "General Retail",
        "Home Improvement",
        "Hobbies",
        "Clothing",
    ],
    "Housing & Bills": [
        "Mortgage/Rent",
        "Phone",
        "Cable/Internet",
        "Service Fees",
        "Utilities",
        "Maintenance/Repairs",
    ],
    "Gifts & Donations": ["Tithing", "Fast Offerings", "Hum Aid", "Gift"],
    "Education": ["Education", "Tutition", "Books"],
    "Entertainment": ["Gaming", "Subscriptions", "Recreation", "Tickets"],
    "Savings & Investments": [
        "Dividends",
        "Investments",
        "Emergency Fund",
        "Retirement",
        "House Down Payment",
    ],
    "Income": [
        "Paychecks",
        "Refunds",
        "CC Rewards",
        "Other Income",
        "Savings / Other Withdrawals",
    ],
    "Miscellaneous": ["Fees & Charges", "Other Services"],
    "Vacations / Travel": ["Airfare", "Travel", "Lodging", "Souvenirs"],
}

FINAL_MAIN_CATEGORY_OPTIONS = sorted(CATEGORY_SUBCATEGORIES)
CATEGORY_PAIR_TO_LABEL = {
    (main_category, sub_category): f"{main_category} :: {sub_category}"
    for main_category, subcategories in CATEGORY_SUBCATEGORIES.items()
    for sub_category in subcategories
}
CATEGORY_LABEL_TO_PAIR = {label: pair for pair, label in CATEGORY_PAIR_TO_LABEL.items()}
CATEGORY_PAIR_OPTIONS = sorted(CATEGORY_LABEL_TO_PAIR)

# Internal-transfer noise kept out of the export on purpose (data cleaning,
# not categorization). These rows appear in the excluded review panel where
# they can be restored.
EXCLUDED_DESCRIPTION_FRAGMENTS = (
    "transfer to sofi",
    "to checking",
    "roundup",
    "home banking transfer",
    "pymt",
    "payment to",
    "internal transfer",
)
PAYMENT_LABEL_PATTERN = re.compile(r"\b(?:payments?|pymt|pmt)\b", re.IGNORECASE)


def _normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def normalize_account_type(value: str | None) -> str:
    """Normalize account labels such as ``CC`` to the established output value."""
    if not value:
        return "unknown"

    normalized = _normalize_label(value)
    known_types = {
        "cc": "credit card",
        "credit": "credit card",
        "creditcard": "credit card",
        "checking": "checking",
        "savings": "savings",
        "imm": "money market",
        "mm": "money market",
        "moneymarket": "money market",
        "gold": "gold account",
        "goldaccount": "gold account",
    }
    return known_types.get(normalized, value.strip().casefold())


def normalize_bank_name(value: str) -> str:
    """Use the bank labels already present in the example categorized CSV."""
    normalized = _normalize_label(value)
    known_banks = {
        "capitalone": "capital one",
        "gwcu": "goldenwest",
        "goldenwest": "goldenwest",
        "goldenwestcreditunion": "goldenwest",
        "sofi": "sofi",
        "usbank": "us bank",
        "robinhood": "robinhood",
    }
    return known_banks.get(normalized, value.strip().casefold())


def normalize_export_filename(value: str) -> str:
    """Return a safe basename with a CSV extension for the download."""
    filename = Path(value.strip()).name or "all_banks_final_categorized.csv"
    path = Path(filename)
    if path.suffix.casefold() != ".csv":
        filename = path.with_suffix(".csv").name if path.suffix else f"{filename}.csv"
    return filename


def validate_final_export_rows(
    transactions: pd.DataFrame,
) -> tuple[pd.DataFrame, str | None]:
    """Drop fully blank editor rows and validate rows before CSV download."""
    missing_columns = set(FINAL_EXPORT_COLUMNS) - set(transactions.columns)
    if missing_columns:
        raise ValueError(f"Missing final export columns: {sorted(missing_columns)}.")

    result = transactions[FINAL_EXPORT_COLUMNS].copy()
    if result.empty:
        return result, "Add at least one complete transaction before downloading."

    blank_rows = result.astype("string").apply(
        lambda column: column.str.strip().eq("").fillna(True)
    ).all(axis=1)
    result = result.loc[~blank_rows].copy()
    if result.empty:
        return result, "Add at least one complete transaction before downloading."

    problems = []
    for column in FINAL_EXPORT_COLUMNS:
        values = result[column].astype("string").str.strip()
        if column == "amount":
            amounts = pd.to_numeric(result[column], errors="coerce")
            if amounts.isna().any():
                problems.append("Amount must be a valid number on every row.")
            else:
                result[column] = amounts
        elif column == "date":
            dates = pd.to_datetime(values, format="%m/%d/%Y", errors="coerce")
            if dates.isna().any():
                problems.append("Date must use MM/DD/YYYY on every row.")
            else:
                result[column] = dates.dt.strftime("%m/%d/%Y")
        elif values.isna().any() or values.eq("").any():
            problems.append(f"{column.replace('_', ' ').title()} is required on every row.")

    if problems:
        return result, " ".join(problems)

    invalid_types = ~result["type"].isin(["debit", "credit"])
    if invalid_types.any():
        problems.append("Type must be debit or credit on every row.")

    invalid_categories = result.apply(
        lambda row: row["sub_category"] not in CATEGORY_SUBCATEGORIES.get(row["main_category"], []),
        axis=1,
    )
    if invalid_categories.any():
        problems.append("Each sub-category must belong to its selected main category.")

    if problems:
        return result, " ".join(problems)
    return result.reset_index(drop=True), None


def edited_row_problem(
    merchant: object,
    main_category: object,
    sub_category: object,
) -> str | None:
    """Return a short problem label for an edited review-table row, or None if valid.

    The merchant must be non-blank, and the sub-category must belong to the
    selected main category (Transfer uses N/A).
    """
    if pd.isna(merchant) or not str(merchant).strip():
        return "missing merchant"
    if main_category not in CATEGORY_SUBCATEGORIES or (
        main_category != "Transfer"
        and sub_category not in CATEGORY_SUBCATEGORIES[main_category]
    ):
        return "sub-category does not match its main category"
    return None


def format_edit_problem_counts(problem_counts: Mapping[str, int]) -> str:
    """Condense per-row edit problems into one short summary sentence."""
    parts = [f"{count} {label}" for label, count in problem_counts.items() if count]
    if not parts:
        return ""
    total = sum(problem_counts.values())
    return f"{total} row(s) need fixes before applying: {', '.join(parts)}."


def get_final_export_row_issues(transactions: pd.DataFrame) -> dict[int, list[str]]:
    """Return validation issues keyed by the row's current zero-based index."""
    issues_by_row: dict[int, list[str]] = {}

    def blank(value: object) -> bool:
        return pd.isna(value) or not str(value).strip()

    for index, row in transactions.iterrows():
        if all(blank(row[column]) for column in FINAL_EXPORT_COLUMNS):
            continue

        issues = []
        date_value = row["date"]
        if blank(date_value) or pd.isna(
            pd.to_datetime(str(date_value).strip(), format="%m/%d/%Y", errors="coerce")
        ):
            issues.append("Date must use MM/DD/YYYY.")

        amount = pd.to_numeric(pd.Series([row["amount"]]), errors="coerce").iloc[0]
        if pd.isna(amount):
            issues.append("Amount must be a valid number.")

        for column in (
            "description", "merchant", "type", "main_category", "sub_category", "bank", "account"
        ):
            if blank(row[column]):
                issues.append(f"{column.replace('_', ' ').title()} is required.")

        if not blank(row["type"]) and row["type"] not in ("debit", "credit"):
            issues.append("Type must be debit or credit.")

        if not blank(row["main_category"]) and not blank(row["sub_category"]):
            if row["sub_category"] not in CATEGORY_SUBCATEGORIES.get(row["main_category"], []):
                issues.append("Sub-category does not belong to the selected main category.")

        if issues:
            issues_by_row[int(index)] = issues
    return issues_by_row


def restored_rows_to_standardized(
    transactions: pd.DataFrame,
    *,
    statuses: pd.Series,
    source_files: pd.Series,
    source_rows: pd.Series,
) -> tuple[pd.DataFrame, str | None]:
    """Validate reviewed rows and convert them back to importer schema."""
    validated, error = validate_final_export_rows(transactions)
    if error:
        return pd.DataFrame(columns=STANDARD_COLUMNS), error

    restored = pd.DataFrame(index=range(len(validated)), columns=STANDARD_COLUMNS)
    restored["transaction_id"] = pd.NA
    restored["date"] = pd.to_datetime(validated["date"], format="%m/%d/%Y", errors="raise").reset_index(drop=True)
    restored["posted_date"] = pd.NaT
    restored["description"] = validated["description"].reset_index(drop=True)
    restored["amount"] = validated["amount"].reset_index(drop=True).astype("float64")
    restored["transaction_type"] = validated["type"].reset_index(drop=True)
    restored["bank_category"] = pd.NA
    restored["status"] = statuses.reset_index(drop=True)
    restored["bank"] = validated["bank"].reset_index(drop=True)
    restored["account"] = validated["account"].reset_index(drop=True)
    restored["currency"] = pd.NA
    restored["source_file"] = source_files.reset_index(drop=True)
    restored["source_row"] = source_rows.reset_index(drop=True)
    return restored, None


def complete_missing_categories(transactions: pd.DataFrame) -> pd.DataFrame:
    """Prepare editor rows for display.

    No categories are auto-assigned anymore - they stay blank until a user
    picks them. The only bookkeeping kept here: a row a user already marked
    as ``Transfer`` gets the ``N/A`` sub-category value.
    """
    result = transactions.copy()
    for index, row in result.iterrows():
        if str(row["main_category"]).strip() == "Transfer":
            result.at[index, "sub_category"] = "N/A"
    return result


def make_transfer_review_rows(
    transactions: pd.DataFrame,
    *,
    source_files: pd.Series,
    source_rows: pd.Series,
) -> pd.DataFrame:
    """Convert user-marked transfers into rows in the excluded-review table."""
    review = transactions[FINAL_EXPORT_COLUMNS].copy().reset_index(drop=True)
    review["main_category"] = "Transfer"
    review["sub_category"] = "N/A"
    review["removal_reason"] = "Manually categorized as Transfer."
    review["status"] = pd.NA
    review["source_file"] = source_files.reset_index(drop=True)
    review["source_row"] = source_rows.reset_index(drop=True)
    review["restore"] = False
    return review[
        FINAL_EXPORT_COLUMNS
        + ["removal_reason", "status", "source_file", "source_row", "restore"]
    ]


def get_exclusion_reason(transaction: pd.Series) -> str | None:
    """Explain why a transaction is omitted from spending totals/export."""
    status_value = transaction.get("status", "")
    status = "" if pd.isna(status_value) else str(status_value)
    if re.search(r"\bdeclined\b", status, re.IGNORECASE):
        return "Bank status is Declined."

    description_value = transaction.get("description", "")
    description = "" if pd.isna(description_value) else str(description_value)
    for fragment in EXCLUDED_DESCRIPTION_FRAGMENTS:
        if fragment in description.casefold():
            return f"Description matches the existing exclusion rule: {fragment}."

    category_value = transaction.get("bank_category", "")
    bank_category = "" if pd.isna(category_value) else str(category_value)
    if PAYMENT_LABEL_PATTERN.search(bank_category):
        return f"Bank category indicates a payment: {bank_category}."

    if PAYMENT_LABEL_PATTERN.search(description):
        return "Description contains a payment label, likely a card-payment transfer."
    return None


def split_filtered_transactions(
    transactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate exportable rows from rows excluded by transaction rules."""
    reasons = transactions.apply(get_exclusion_reason, axis=1)
    excluded_mask = reasons.notna()
    included = transactions.loc[~excluded_mask].copy()
    excluded = transactions.loc[excluded_mask].copy()
    excluded["removal_reason"] = reasons.loc[excluded_mask]
    return included, excluded


def split_duplicate_transactions(
    transactions: pd.DataFrame,
    existing_transactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep the first exact transaction and return later copies for review."""
    if transactions.empty:
        return transactions.copy(), transactions.assign(removal_reason=pd.Series(dtype="string"))

    key_columns = {"date", "description", "amount", "bank", "account"}
    if not key_columns.issubset(transactions.columns):
        raise ValueError(f"Duplicate detection requires columns: {sorted(key_columns)}.")

    def transaction_key(row: pd.Series) -> tuple | None:
        transaction_date = pd.to_datetime(row["date"], errors="coerce")
        amount = pd.to_numeric(pd.Series([row["amount"]]), errors="coerce").iloc[0]
        if pd.isna(transaction_date) or pd.isna(amount):
            return None
        description = " ".join(str(row["description"]).split()).casefold()
        if not description:
            return None
        return (
            transaction_date.date(),
            description,
            float(amount),
            normalize_bank_name(str(row["bank"])),
            normalize_account_type(str(row["account"])),
        )

    seen = set()
    if not existing_transactions.empty and key_columns.issubset(existing_transactions.columns):
        for _, row in existing_transactions.iterrows():
            key = transaction_key(row)
            if key is not None:
                seen.add(key)

    duplicate_indices = []
    for index, row in transactions.iterrows():
        key = transaction_key(row)
        if key is not None and key in seen:
            duplicate_indices.append(index)
        elif key is not None:
            seen.add(key)

    duplicate_mask = transactions.index.isin(duplicate_indices)
    included = transactions.loc[~duplicate_mask].copy()
    duplicates = transactions.loc[duplicate_mask].copy()
    duplicates["removal_reason"] = (
        "Duplicate entry: exact transaction already imported for this date, bank, and account."
    )
    return included, duplicates


def format_final_export(
    transactions: pd.DataFrame,
    *,
    apply_exclusions: bool = True,
) -> pd.DataFrame:
    """Create the same nine-column categorized schema as the example CSV.

    No rules, caches, or defaults are applied: ``main_category`` and
    ``sub_category`` start blank for the user (or the ML models in
    ``ml_models.py``) to fill in, and ``merchant`` is just the cleaned
    description unless a user edits it.
    """
    required_columns = {"date", "description", "amount", "transaction_type", "bank", "account"}
    missing_columns = required_columns - set(transactions.columns)
    if missing_columns:
        raise ValueError(f"Missing standardized transaction columns: {sorted(missing_columns)}.")

    if apply_exclusions:
        result, _ = split_filtered_transactions(transactions)
    else:
        result = transactions.copy()
    if "merchant" not in result:
        result["merchant"] = result["description"].map(
            lambda description: smart_title(str(description))
        )
    else:
        missing_merchants = result["merchant"].isna() | result["merchant"].astype("string").str.strip().eq("")
        result.loc[missing_merchants, "merchant"] = result.loc[
            missing_merchants, "description"
        ].map(lambda description: smart_title(str(description)))

    result["type"] = result["transaction_type"]
    result["bank"] = result["bank"].map(normalize_bank_name)
    result["account"] = result["account"].map(normalize_account_type)
    result["main_category"] = pd.Series(index=result.index, dtype="string")
    result["sub_category"] = pd.Series(index=result.index, dtype="string")
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result = result.sort_values("date", kind="stable")
    result["date"] = result["date"].dt.strftime("%m/%d/%Y")
    return result[FINAL_EXPORT_COLUMNS].copy()