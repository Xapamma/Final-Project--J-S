"""
Build one cleaned, unlabeled transaction CSV per person.

Expected input layout:
    .data/<person>/<bank>/<account>.csv

The headless pipeline is independent from the Streamlit upload workflow. The
Streamlit app can still import individual statements and export its own
session data.

Duplicates are flagged, never removed automatically. Exact repeats (same date,
raw description, amount, bank, account) are written to
    .data/<person>/duplicates_review.csv
with a blank `decision` column. Type `remove` (keep one copy, drop the rest) or
`keep` (they are real repeat purchases) and rerun. Blank = keep everything until
someone decides. Decisions are remembered between runs.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from dashboard.cleaning_logic import clean_description, smart_title

DATA_FOLDER = ".data"
PEOPLE = {
    "savanna": "savanna_transactions.csv",
    "jillian": "jillian_transactions.csv",
}
KNOWN_BANKS = ["goldenwest", "sofi", "capital one", "bank of america"]
ACCOUNT_MAP = {
    "cc": "credit card",
    "credit card": "credit card",
    "checking": "checking",
    "savings": "savings",
    "imm": "money market",
    "gold": "gold account",
}
NOISE_PATTERNS = [
    "transfer to sofi",
    "to checking",
    "angel funding",
    "roundup",
    "home banking transfer",
    "pymt",
    "payment to",
    "north capital",
    "internal transfer",
]
DUPLICATE_REVIEW_FILE = "duplicates_review.csv"
DUPLICATE_KEY = ["date", "raw_description", "amount", "bank", "account"]
DUPLICATE_DECISIONS = {"keep", "remove"}
OUTPUT_COLUMNS = [
    "date",
    "description",
    "raw_description",
    "type",
    "amount",
    "main_category",
    "sub_category",
    "bank",
    "account",
]


def load_bank_file(file: str, bank_name: str) -> pd.DataFrame:
    """Read one bank CSV and convert it to the common column layout."""
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower()

    if bank_name == "goldenwest":
        df = df.rename(
            columns={
                "posting date": "date",
                "type": "category type",
            }
        )
    elif bank_name == "capital one":
        df = df.rename(columns={"transaction date": "date"})
        df = pd.melt(
            df,
            id_vars=[c for c in df.columns if c not in ["debit", "credit"]],
            value_vars=["debit", "credit"],
            var_name="type",
            value_name="amount",
        ).dropna(subset=["amount"])
    elif bank_name == "bank of america":
        # BofA credit-card export: Transaction Date, Posting Date, Description,
        # Reference Number, Amount (always positive), Type (debit/credit), Statement.
        # The sign is set below from the `type` column.
        df = df.rename(columns={"transaction date": "date"})

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["amount"] = (
        df["amount"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("(", "-", regex=False)
        .str.replace(")", "", regex=False)
        .astype(float)
    )

    filename_lower = os.path.basename(file).lower()
    matched_key = process.extractOne(
        filename_lower, list(ACCOUNT_MAP), scorer=fuzz.partial_ratio
    )[0]
    account_name = ACCOUNT_MAP[matched_key]
    df["bank"] = bank_name
    df["account"] = account_name

    if account_name == "credit card":
        if "type" in df:
            transaction_type = df["type"]
        elif "category type" in df:
            transaction_type = df["category type"]
        else:
            raise ValueError(f"{file} is missing a debit/credit type column.")
        df["type"] = transaction_type.astype(str).str.casefold()
        df.loc[df["type"] == "debit", "amount"] = -df["amount"].abs()
        df.loc[df["type"] == "credit", "amount"] = df["amount"].abs()
    else:
        df["type"] = np.where(df["amount"] > 0, "credit", "debit")

    return df


def _duplicate_key_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalized key columns so reruns and the review CSV compare equal."""
    return pd.DataFrame({
        "date": pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d"),
        "raw_description": df["raw_description"].astype(str).str.strip(),
        "amount": pd.to_numeric(df["amount"], errors="coerce").round(2),
        "bank": df["bank"].astype(str),
        "account": df["account"].astype(str),
    }, index=df.index)


def review_duplicates(combined: pd.DataFrame, person_folder: str) -> pd.DataFrame:
    """Flag exact repeat transactions; drop extra copies only when approved.

    Writes/updates <person_folder>/duplicates_review.csv with one row per
    group of identical transactions. Only groups whose `decision` is
    `remove` lose their extra copies (the first copy is kept).
    """
    keys = _duplicate_key_frame(combined)
    is_dup = keys.duplicated(keep=False)
    review_path = os.path.join(person_folder, DUPLICATE_REVIEW_FILE)

    # Earlier decisions, keyed by the duplicate key.
    decisions: dict[tuple, str] = {}
    if os.path.exists(review_path):
        previous = pd.read_csv(review_path, keep_default_na=False, dtype=str)
        previous_keys = _duplicate_key_frame(previous)
        for key, decision in zip(previous_keys.itertuples(index=False, name=None),
                                 previous["decision"].str.strip().str.casefold()):
            if decision:
                if decision not in DUPLICATE_DECISIONS:
                    raise ValueError(
                        f"{review_path}: decision must be blank, 'keep' or 'remove', got {decision!r}."
                    )
                decisions[key] = decision

    if not is_dup.any():
        if os.path.exists(review_path):
            os.remove(review_path)
        print("  Duplicates: none found")
        return combined

    groups = (
        keys.loc[is_dup]
        .assign(source_file=combined.loc[is_dup, "source_file"])
        .groupby(DUPLICATE_KEY, sort=True, dropna=False)
        .agg(copies=("source_file", "size"),
             source_files=("source_file", lambda f: "; ".join(sorted(set(f)))))
        .reset_index()
    )
    groups["found_in"] = np.where(groups["source_files"].str.contains(";"),
                                  "different files", "same file")
    groups["decision"] = [
        decisions.get(key, "")
        for key in groups[DUPLICATE_KEY].itertuples(index=False, name=None)
    ]
    groups.to_csv(review_path, index=False)

    remove_keys = set(
        groups.loc[groups["decision"] == "remove", DUPLICATE_KEY]
        .itertuples(index=False, name=None)
    )
    row_keys = pd.Series(list(keys.itertuples(index=False, name=None)), index=keys.index)
    extra_copy = keys.duplicated(keep="first") & row_keys.isin(remove_keys)
    pending = int((groups["decision"] == "").sum())

    print(
        f"  Duplicates: {len(groups)} group(s) flagged -> {review_path}\n"
        f"    removed {int(extra_copy.sum())} approved extra copies, "
        f"{pending} group(s) waiting for a decision (kept for now)"
    )
    return combined.loc[~extra_copy].copy()


def process_person(person_folder: str) -> pd.DataFrame | None:
    """Load, combine, clean, and return every bank CSV for one person."""
    bank_frames: list[pd.DataFrame] = []
    bank_folders = sorted(
        entry.path for entry in os.scandir(person_folder) if entry.is_dir()
    )

    for bank_folder in bank_folders:
        raw_bank_name = os.path.basename(bank_folder)
        bank_name, bank_score, _ = process.extractOne(
            raw_bank_name, KNOWN_BANKS, scorer=fuzz.ratio
        )
        print(
            f"  Bank folder: {raw_bank_name} -> matched to "
            f"{bank_name} (score {bank_score})"
        )
        files = sorted(glob.glob(os.path.join(bank_folder, "*.csv")))
        if not files:
            print(f"  (no CSV files in {bank_folder}, skipping)")
            continue
        bank_frames.extend(
            load_bank_file(file, bank_name).assign(source_file=os.path.basename(file))
            for file in files
        )

    if not bank_frames:
        return None

    combined = pd.concat(bank_frames, ignore_index=True).sort_values("date")
    combined["description"] = combined["description"].fillna("")
    combined["raw_description"] = combined["description"]

    descriptions = combined["description"].str.lower()
    is_noise = descriptions.apply(
        lambda value: any(pattern in value for pattern in NOISE_PATTERNS)
    )
    combined = combined.loc[~is_noise].copy()
    combined = review_duplicates(combined, person_folder)
    combined["description"] = combined["description"].apply(clean_description).apply(
        smart_title
    )
    combined["main_category"] = np.nan
    combined["sub_category"] = np.nan
    return combined[OUTPUT_COLUMNS]


def main() -> None:
    """Build the configured per-person output files."""
    written: list[str] = []
    for person, output_csv in PEOPLE.items():
        person_folder = os.path.join(DATA_FOLDER, person)
        print(f"\nProcessing {person} ({person_folder})")
        if not os.path.isdir(person_folder):
            print(
                f"  Folder not found - skipping {person}. "
                f"Put bank folders in {person_folder}{os.sep}<bank>{os.sep}"
            )
            continue

        transactions = process_person(person_folder)
        if transactions is None:
            print(f"  No bank CSVs found for {person} - nothing written.")
            continue

        transactions.to_csv(output_csv, index=False)
        written.append(output_csv)
        print(f"  Saved {len(transactions)} rows -> {output_csv}")

    if written:
        print("\nDone! Cleaned, unlabeled datasets saved:", ", ".join(written))
    else:
        print(
            "\nNo output written. Expected layout: "
            ".data/<person>/<bank>/*.csv"
        )


if __name__ == "__main__":
    main()
