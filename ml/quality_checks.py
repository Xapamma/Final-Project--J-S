"""
quality_checks.py
=================
Data-checkpoint quality checks for one person's processed transactions.

Reports ONLY aggregate numbers (no descriptions), so the output file is safe
to commit:
    - rows per source file / bank / account, and date range
    - rows changed by the noise filter
    - possible duplicates, missing values, date gaps
    - transactions per main category (labeled vs. unlabeled)

Run from the project root:
    uv run python -m ml.quality_checks jillian
    uv run python -m ml.quality_checks jillian --labeled jillian_transactions_labeled.csv
"""

import argparse
import glob
import os

import pandas as pd

from ml.bank_statment_processor import DATA_FOLDER, NOISE_PATTERNS, PEOPLE

RESULTS_DIR = "results"


def _md_table(df):
    return df.to_markdown() if hasattr(df, "to_markdown") and _has_tabulate() else "```\n" + df.to_string() + "\n```"


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def raw_file_summary(person):
    """Row counts per raw CSV in .data/<person>/<bank>/ (before any cleaning)."""
    rows = []
    # Statements with no CSV export are entered by hand into
    # .data/manual/<person>_*transcribed*.csv and count as raw sources too.
    paths = sorted(glob.glob(os.path.join(DATA_FOLDER, person, "*", "*.csv")))
    paths += sorted(glob.glob(os.path.join(DATA_FOLDER, "manual", f"{person}_*transcribed*.csv")))
    for path in paths:
        raw = pd.read_csv(path)
        raw.columns = raw.columns.str.strip().str.lower()
        date_col = next((c for c in raw.columns if "date" in c), None)
        dates = pd.to_datetime(raw[date_col], errors="coerce") if date_col else pd.Series(dtype="datetime64[ns]")
        rows.append({
            "bank_folder": os.path.basename(os.path.dirname(path)),
            "file": os.path.basename(path),
            "raw_rows": len(raw),
            "first_date": dates.min().date() if dates.notna().any() else None,
            "last_date": dates.max().date() if dates.notna().any() else None,
            "downloaded_(file_mtime)": pd.Timestamp(os.path.getmtime(path), unit="s").date(),
        })
    return pd.DataFrame(rows)


def run_checks(person, labeled_csv=None, out=None):
    processed_csv = PEOPLE[person]
    df = pd.read_csv(labeled_csv or processed_csv, parse_dates=["date"],
                     keep_default_na=False, na_values=[""])  # "N/A" is a real sub-category
    raw_files = raw_file_summary(person)
    raw_total = int(raw_files["raw_rows"].sum()) if not raw_files.empty else 0

    has_raw = "raw_description" in df.columns
    built_by = ("`ml/bank_statment_processor.py`" if has_raw
                else "the Streamlit app export (no `raw_description`, so source-step counts are unavailable)")
    lines = [f"# Data quality checks - {person}", "",
             f"Source: `{labeled_csv or processed_csv}` (built by {built_by})", ""]

    # 1. Sources
    lines += ["## Source files", "", _md_table(raw_files.set_index("file")) if not raw_files.empty else "(raw files not available on this machine)", ""]

    # 2. Rows changed by the noise filter (needs raw files + raw_description)
    if has_raw and raw_total:
        raw_desc = df["raw_description"].fillna("").astype(str)
        noise_left = raw_desc.str.lower().apply(lambda x: any(n in x for n in NOISE_PATTERNS)).sum()
        steps = [
            {"step": "raw rows loaded", "rows": raw_total},
            {"step": "dropped by noise/transfer filter (and approved duplicate removal)", "rows": raw_total - len(df)},
            {"step": "rows after cleaning", "rows": len(df)},
        ]
        steps.append({"step": "noise-pattern rows remaining (should be 0)", "rows": int(noise_left)})
        steps = pd.DataFrame(steps).set_index("step")
        lines += ["## Cleaning steps", "", _md_table(steps), ""]
    else:
        lines += ["## Cleaning steps", "", "Not available: needs the raw bank files and the processor's `raw_description` column.", ""]

    # 3. Basic quality
    text_col = "raw_description" if has_raw else "description"
    dup_cols = ["date", text_col, "amount", "bank", "account"]
    dups = int(df.duplicated(subset=dup_cols).sum())
    missing = df.isna().sum().rename("missing").to_frame()
    lines += ["## Duplicates and missing values", "",
              f"- Possible duplicates (same date, {text_col.replace('_', ' ')}, amount, bank, account): **{dups}** "
              "(could be real repeat purchases; the processor flags these for review)", "",
              _md_table(missing), ""]

    # 4. Coverage / date gaps
    by_acct = df.groupby(["bank", "account"]).agg(
        rows=("amount", "size"), first_date=("date", "min"), last_date=("date", "max"))
    by_acct["first_date"] = by_acct["first_date"].dt.date
    by_acct["last_date"] = by_acct["last_date"].dt.date
    monthly = df.groupby(df["date"].dt.to_period("M")).size().rename("transactions").to_frame()
    all_months = pd.period_range(df["date"].min(), df["date"].max(), freq="M")
    empty_months = [str(m) for m in all_months if m not in monthly.index]
    gaps = df["date"].sort_values().diff().dt.days
    lines += ["## Coverage", "", _md_table(by_acct), "",
              f"- Longest gap between transactions: **{int(gaps.max())} days**",
              f"- Months with no transactions: {', '.join(empty_months) or 'none'}", "",
              _md_table(monthly), ""]

    # 5. Labels
    labeled_mask = df["main_category"].notna()
    lines += ["## Labels", "",
              f"- Labeled rows: **{int(labeled_mask.sum())} / {len(df)}**", ""]
    if labeled_mask.any():
        per_cat = (df.loc[labeled_mask]
                   .groupby("main_category").size().sort_values(ascending=False)
                   .rename("transactions").to_frame())
        per_cat["share"] = (per_cat["transactions"] / per_cat["transactions"].sum()).round(3)
        lines += [_md_table(per_cat), "",
                  f"- Distinct sub-categories used: {df.loc[labeled_mask, 'sub_category'].nunique()}",
                  f"- Distinct descriptions: {df[text_col].nunique()}", ""]

    report = "\n".join(lines)
    out = out or os.path.join(RESULTS_DIR, f"data_quality_{person}.md")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write(report)
    print(report)
    print(f"\nSaved -> {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("person", choices=sorted(PEOPLE))
    parser.add_argument("--labeled", help="labeled CSV to use instead of the processor output")
    parser.add_argument("--output", help="where to write the report (default results/data_quality_<person>.md)")
    args = parser.parse_args()
    run_checks(args.person, args.labeled, args.output)
