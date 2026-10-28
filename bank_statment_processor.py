"""
bank_statment_processor.py
==========================
Loads every raw bank-statement CSV inside `.data/`, normalizes it into a
single common format, and saves one combined dataset to
`all_banks_final_categorized.csv`.

WHAT CHANGED (2026-10-27) - rule-based categorization REMOVED
--------------------------------------------------------------
* No more description_map / sub_category_map / category_hierarchy.
* No more merchant_cache.json lookups.
* No more "General Spending" / "Other Income" / "Investments" defaults.

The output file now has EMPTY `main_category` / `sub_category` columns on
purpose. WE pick the categories for each transaction (by hand in the CSV,
or later with our ML models in `ml_models.py`).

Once `ml_models.py` is implemented, categorize everything with:
    python -c "from ml_models import train_and_categorize; train_and_categorize()"
"""

import pandas as pd
import numpy as np
import glob
import os
from rapidfuzz import process, fuzz

from cleaning_logic import clean_description, smart_title

# Root folder containing one sub-folder per bank
folder = ".data"

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

# List to hold all banks
all_banks = []

# Scan folders in .data
bank_folders = [f.path for f in os.scandir(folder) if f.is_dir()]

for bank_folder in bank_folders:
    raw_bank_name = os.path.basename(bank_folder)

    # Fuzzy match bank name
    match_result = process.extractOne(raw_bank_name, known_banks, scorer=fuzz.ratio)
    bank_name = match_result[0]
    bank_score = match_result[1]
    print(f"Processing bank folder: {raw_bank_name} -> matched to {bank_name} (score {bank_score})")

    # Find CSV files in the bank folder
    files = glob.glob(os.path.join(bank_folder, "*.csv"))
    bank_dfs = []

    for file in files:
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
        matched_key = match_result[0]
        account_name = account_map[matched_key]

        df["bank"] = bank_name
        df["account"] = account_name

        # Add a "type" column based on the amount
        # If credit card, make debits negative
        if account_name == "credit card":
            df.loc[df["type"] == "debit", "amount"] = -df["amount"].abs()   # charges negative
            df.loc[df["type"] == "credit", "amount"] = df["amount"].abs()   # payments positive
        else:
            df["type"] = np.where(df["amount"] > 0, "credit", "debit")

        bank_dfs.append(df)

    # Combine all accounts for this bank
    bank_combined = pd.concat(bank_dfs, ignore_index=True)
    all_banks.append(bank_combined)

# Combine all banks
combined = pd.concat(all_banks, ignore_index=True)
combined = combined.sort_values("date")

# Don't lose the original bank text - keep it around for reference and as
# an alternative text feature for our ML models (raw vs cleaned comparison).
combined["description"] = combined["description"].fillna("")
combined["raw_description"] = combined["description"]

# --- 1. Drop internal noise ------------------------------------------------
# Transfers / roundups between our own accounts are neither spending nor
# income, so they would only add noise to the dataset.
noise_patterns = [
    "transfer to sofi", "to checking", "roundup",
    "home banking transfer", "pymt", "payment to", 
    "internal transfer",
]
desc_lower = combined["description"].str.lower()
mask = ~desc_lower.apply(lambda x: any(noise in x for noise in noise_patterns))
combined = combined[mask].copy()

# --- 2. Clean descriptions (display text + ML text feature) ----------------
combined["description"] = combined["description"].apply(clean_description).apply(smart_title)

# --- 3. Categories are chosen by US (by hand or by ML) ---------------------
# No auto-categorization anymore. These columns start empty on purpose:
#   * fill them in yourself (that is the labeling step -> ML training data)
#   * or run the ML models in ml_models.py to fill them automatically
combined["main_category"] = np.nan
combined["sub_category"] = np.nan

# Keep only necessary columns
columns_to_keep = [
    "date", "description", "raw_description", "type", "amount",
    "main_category", "sub_category", "bank", "account",
]
combined = combined[columns_to_keep]

# Save
combined.to_csv("all_banks_final_categorized.csv", index=False)

print()
print("Done! Cleaned, UNLABELED dataset saved as 'all_banks_final_categorized.csv'.")
print("main_category / sub_category are blank on purpose - pick your own labels,")
print("then run the ML models in ml_models.py to categorize every transaction.")
print("Example (after implementing ml_models.py):")
print('  python -c "from ml_models import train_and_categorize; train_and_categorize()"')