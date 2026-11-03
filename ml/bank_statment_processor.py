"""
bank_statment_processor.py
==========================
Loads every raw bank-statement CSV inside `.data/`, normalizes it into a
single common format, and saves ONE CSV PER PERSON:

    .data/savanna/<bank>/<account>.csv  ->  savanna_transactions.csv  (Person A, training)
    .data/jillian/<bank>/<account>.csv  ->  jillian_transactions.csv  (Person B, testing)

Both files go through exactly the same cleaning, so they have identical
columns and can be compared fairly (train on one, test on the other).

WHAT CHANGED (2026-11-02) - one output file per person
-------------------------------------------------------
* Raw data now lives in a sub-folder per person, then per bank.
* Instead of one combined CSV we write one CSV per person, so the file a
  row is in tells us whose transaction it is (no `person` column needed).

WHAT CHANGED (2026-10-27) - rule-based categorization REMOVED
--------------------------------------------------------------
* No more description_map / sub_category_map / category_hierarchy.
* No more merchant_cache.json lookups.
* No more "General Spending" / "Other Income" / "Investments" defaults.

The output files have EMPTY `main_category` / `sub_category` columns on
purpose. WE pick the categories for each transaction (by hand in the CSV,
or later with our ML models in `ml/ml_models.py`).

Run from the project root:
    uv run python -m ml.bank_statment_processor
"""

import pandas as pd
import numpy as np
import glob
import os
from rapidfuzz import process, fuzz

from dashboard.cleaning_logic import clean_description, smart_title

# Root folder containing one sub-folder per person, each with one sub-folder per bank
DATA_FOLDER = ".data"

# Person folder name -> output CSV
PEOPLE = {
    "savanna": "savanna_transactions.csv",  # Person A (training)
    "jillian": "jillian_transactions.csv",  # Person B (testing)
}

# Known banks and account types
known_banks = ["goldenwest", "sofi", "capital one"]
account_map = {
    "cc": "credit card",
    "credit card": "credit card",
    "checking": "checking",
    "savings": "savings",
    "imm": "money market",
    "gold": "gold account",
}

# Transfers / roundups between our own accounts are neither spending nor
# income, so they would only add noise to the dataset.
noise_patterns = [
    "transfer to sofi", "to checking", "angel funding", "roundup",
    "home banking transfer", "pymt", "payment to", "north capital",
    "internal transfer",
]

columns_to_keep = [
    "date", "description", "raw_description", "type", "amount",
    "main_category", "sub_category", "bank", "account",
]


def load_bank_file(file, bank_name):
    """Read one bank CSV and convert it to the common column layout."""
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip().str.lower()

    # Bank-specific column cleaning
    if bank_name == "goldenwest":
        df = df.rename(columns={
            "posting date": "date",
            "description": "description",
            "amount": "amount",
            "type": "category type",
        })

    elif bank_name == "capital one":
        df = df.rename(columns={
            "transaction date": "date",
            "description": "description",
        })
        # Debit/credit live in two columns -> melt them into one amount column
        df = pd.melt(
            df,
            id_vars=[c for c in df.columns if c not in ["debit", "credit"]],
            value_vars=["debit", "credit"],
            var_name="type",
            value_name="amount",
        ).dropna(subset=["amount"])

    elif bank_name == "sofi":
        # Typical SoFi CSV export columns
        df = df.rename(columns={
            "date": "date",
            "description": "description",
            "amount": "amount",
        })

    # Convert date column
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Clean amount column
    df["amount"] = (
        df["amount"]
        .astype(str)
        .str.replace(",", "")
        .str.replace("(", "-")
        .str.replace(")", "")
        .astype(float)
    )

    # Fuzzy match account type from filename
    filename_lower = os.path.basename(file).lower()
    match_result = process.extractOne(filename_lower, account_map.keys(), scorer=fuzz.partial_ratio)
    account_name = account_map[match_result[0]]

    df["bank"] = bank_name
    df["account"] = account_name

    # Add a "type" column based on the amount
    # If credit card, make debits negative
    if account_name == "credit card":
        df.loc[df["type"] == "debit", "amount"] = -df["amount"].abs()   # charges negative
        df.loc[df["type"] == "credit", "amount"] = df["amount"].abs()   # payments positive
    else:
        df["type"] = np.where(df["amount"] > 0, "credit", "debit")

    return df


def process_person(person_folder):
    """Load, combine, and clean every bank CSV for one person."""
    all_banks = []

    bank_folders = sorted(f.path for f in os.scandir(person_folder) if f.is_dir())
    for bank_folder in bank_folders:
        raw_bank_name = os.path.basename(bank_folder)

        # Fuzzy match bank name
        bank_name, bank_score, _ = process.extractOne(raw_bank_name, known_banks, scorer=fuzz.ratio)
        print(f"  Bank folder: {raw_bank_name} -> matched to {bank_name} (score {bank_score})")

        files = sorted(glob.glob(os.path.join(bank_folder, "*.csv")))
        if not files:
            print(f"  (no CSV files in {bank_folder}, skipping)")
            continue
        all_banks.extend(load_bank_file(file, bank_name) for file in files)

    if not all_banks:
        return None

    combined = pd.concat(all_banks, ignore_index=True)
    combined = combined.sort_values("date")

    # Don't lose the original bank text - keep it around for reference and as
    # an alternative text feature for our ML models (raw vs cleaned comparison).
    combined["description"] = combined["description"].fillna("")
    combined["raw_description"] = combined["description"]

    # --- 1. Drop internal noise --------------------------------------------
    desc_lower = combined["description"].str.lower()
    mask = ~desc_lower.apply(lambda x: any(noise in x for noise in noise_patterns))
    combined = combined[mask].copy()

    # --- 2. Clean descriptions (display text + ML text feature) ------------
    combined["description"] = combined["description"].apply(clean_description).apply(smart_title)

    # --- 3. Categories are chosen by US (by hand or by ML) -----------------
    combined["main_category"] = np.nan
    combined["sub_category"] = np.nan

    return combined[columns_to_keep]


def main():
    written = []
    for person, output_csv in PEOPLE.items():
        person_folder = os.path.join(DATA_FOLDER, person)
        print(f"\nProcessing {person} ({person_folder})")

        if not os.path.isdir(person_folder):
            print(f"  Folder not found - skipping {person}. "
                  f"Put their bank folders in {person_folder}/<bank>/<account>.csv")
            continue

        df = process_person(person_folder)
        if df is None:
            print(f"  No bank CSVs found for {person} - nothing written.")
            continue

        df.to_csv(output_csv, index=False)
        written.append(output_csv)
        print(f"  Saved {len(df)} rows -> {output_csv}")

    print()
    if written:
        print("Done! Cleaned, UNLABELED datasets saved:", ", ".join(written))
        print("main_category / sub_category are blank on purpose - pick your own labels,")
        print("then run the ML models in ml/ml_models.py.")
    else:
        print("No output written. Expected layout: .data/savanna/<bank>/*.csv and .data/jillian/<bank>/*.csv")


if __name__ == "__main__":
    main()
