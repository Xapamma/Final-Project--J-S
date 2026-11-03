"""
Build one cleaned, unlabeled transaction CSV per person.

Expected input layout:
    .data/<person>/<bank>/<account>.csv

The headless pipeline is independent from the Streamlit upload workflow. The
Streamlit app can still import individual statements and export its own
session data.
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
KNOWN_BANKS = ["goldenwest", "sofi", "capital one"]
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
        bank_frames.extend(load_bank_file(file, bank_name) for file in files)

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
