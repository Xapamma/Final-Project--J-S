"""
cleaning_logic.py
=================
Description CLEANING helpers only.

The old rule-based categorization system (description_map,
sub_category_map, category_hierarchy, merchant cache, and the
"General Spending" / "Other Income" / "Investments" defaults) has been
REMOVED so we can label transactions ourselves and train our own ML
models (see ml_models.py).

This module only keeps the pieces that make bank descriptions easier to
read and easier for a model to learn from.
"""

import re


def smart_title(text):
    """Title-case a string while keeping apostrophes intact.

    Example: 'WAL-MART SUPERCENTER #1234 UT' -> 'Wal-Mart Supercenter #1234 Ut'
    """
    words = str(text).lower().split()
    fixed = []
    for w in words:
        if "'" in w:
            first, rest = w.split("'", 1)
            fixed.append(first.capitalize() + "'" + rest.lower())
        else:
            fixed.append(w.capitalize())
    return " ".join(fixed)


def clean_description(description):
    """Lowercase a bank description and strip bank noise, digits, and symbols.

    The original bank text is kept as `raw_description` in the dataset,
    while this cleaned text is the main feature we feed our ML models.
    """
    raw_desc = str(description).lower()

    # Normalize a bare "withdrawal" line
    raw_desc = re.sub(r"^\s*withdrawal\s*$", "bank withdrawal", raw_desc)

    # Drop common bank prefixes / location / device noise
    raw_desc = re.sub(
        r"^withdrawal\s*(xx)?\s*x?\b|xx\s*(sq|card|[a-z])?|^recurring\s*withdrawal\s*|provo\s*ut|\borem\b",
        "",
        raw_desc,
    )

    # Remove numbers + symbols (store numbers, phone digits, etc.)
    raw_desc = re.sub(r"[^a-z&'\-\s]", "", raw_desc)

    # Normalize spacing
    raw_desc = " ".join(raw_desc.split())

    return raw_desc


# Alias used by the Streamlit dashboard modules (kept from the old repo name).
# It is the same description-cleaning helper - no categorization logic.
clean_description_for_matching = clean_description