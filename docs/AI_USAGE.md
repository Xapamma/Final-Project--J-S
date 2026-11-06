# AI Use Disclosure & Log

STAT 486 Fall 2026 – Savanna McLelland & Jillian Baker
Updated: October 27, 2026

The project plan promises that "we will … disclose any AI help in `report.md`. "
This file is the detailed, dated log behind that disclosure. A short summary
also lives in `report.md` (§ 7).

## Policy

* We may use AI as a **pair-programming assistant** (writing skeleton code,
  explaining errors, suggesting tests/documentation), but **we review and
  understand every change before we commit it**.
* The ML experiments (what gets trained, which method wins, what it means) are
  **ours** — we design the experiments, review the numbers, and own the
  conclusions. AI is never allowed to silently make project decisions for us.
* Every AI contribution gets a dated entry below, and teammates review each
  other's code as stated in the project plan.
* We will confirm this policy with the instructor and adapt it if the class or
  instructor policy is stricter.

> If an instructor or class policy overrides anything in this file, this
> section wins: **follow the class policy** and update this log accordingly.

## What AI has done so far

| Date | What AI did | Where | Reviewed by |
|------|-------------|-------|-------------|
| 2026-10-27 | Replaced the rule-based categorizer with a manual-labeling workflow: categories now start blank in the scripts and the Streamlit app (now our labeling tool); scaffolded `ml/ml_models.py` (TODO stubs only, no training code written); wrote tests (31 passing) and the docs. | `ml/bank_statment_processor.py`, `dashboard/cleaning_logic.py`, `dashboard.py`, `app.py`, `ml/ml_models.py`, `tests/`, `docs/` | Savanna & Jillian |
| 2026-10-27 | Reorganized the repo for clarity: moved project docs into `docs/`, shared dashboard modules into the `dashboard/` package, headless pipeline + ML scripts into `ml/`; removed the redundant `bank_data_analytics.py`; added `.gitignore`; stopped tracking private data (`.data/`, `all_banks_final_categorized.csv`) and `__pycache__`. | repo layout: `docs/`, `dashboard/`, `ml/`, `tests/`, `.gitignore` | Savanna & Jillian |
| 2026-11-02 | Added an optional resume flow that loads a prior website CSV download, preserves existing category labels, and lets users continue labeling uncategorized rows in a new session. | `app.py`, `dashboard/final_export.py`, `tests/` | Savanna & Jillian |
| 2026-11-02 | Normalized bank labels when resuming a prior download and restored the combined Main :: Sub category selector in one-by-one transaction review. | `app.py`, `dashboard/final_export.py`, `tests/` | Savanna & Jillian |
| 2026-11-02 | Reset review-panel state and widget keys when loading a continuation CSV so the new unresolved rows can be selected and applied immediately. | `app.py` | Savanna & Jillian |
| 2026-11-02 | Improved the review toolbar visibility and made selecting a row in the read-only transaction table enter edit mode. | `app.py` | Savanna & Jillian |
| 2026-11-02 | Changed the read-only table trigger to single-cell selection so clicking directly into any cell opens the editable table with category dropdown options. | `app.py` | Savanna & Jillian |
| 2026-11-02 | Fixed the sidebar bank count to combine normalized bank labels from imported and editable data, and refresh immediately after adding a statement. | `app.py` | Savanna & Jillian |
| 2026-11-02 | Moved the excluded-transaction restore panel to the beginning of the review workflow for easier access. | `app.py` | Savanna & Jillian |
| 2026-11-02 | Moved the Restore checkbox column to the first position in the excluded-transaction review table. | `app.py` | Savanna & Jillian |
| 2026-11-02 | Placed Why it was removed immediately after the Restore checkbox in the excluded-transaction review table. | `app.py` | Savanna & Jillian |
| 2026-11-02 | Removed the unused merchant field from the export schema, review interfaces, analytics views, and category-edit workflow. | `app.py`, `dashboard/`, `pages/`, `tests/` | Savanna & Jillian |
| 2026-11-02 | Fixed an empty Markdown call left by the merchant removal that prevented the add-category selector from rendering. | `app.py` | Savanna & Jillian |
| 2026-11-02 | Removed the Apply to / All matching column and its bulk category propagation so edits apply only to the selected transaction row. | `app.py` | Savanna & Jillian |

Notes on what the assistant saw/touched:

* During development the assistant read and processed the **local** bank
  CSVs in `.data/` to build and validate the pipeline. Nothing was uploaded to
  any external service; all runs happened locally in this workspace.
* The processed `all_banks_final_categorized.csv` in this workspace contains
  real transactions, so it must stay out of shared/committed copies per our
  privacy plan. A made-up sample will be added later for reproduction.

## Future uses (as of today)

These are planned uses, not promises — every actual use gets a dated entry
above before we rely on it:

1. **Implementing the ML methods in `ml_models.py`.** We will likely ask AI to
   help write the scikit-learn code (vectorizer, KNN, Random Forest, ANN, train/test
   split, confusion matrix). Rule: we approve each function, run the tests, and
   sanity-check results against our most-common baseline before trusting any number.
   The final code stays attributed to both of us, and the AI part is disclosed here.
2. **Debugging and refactoring** (tracebacks, pandas/Streamlit quirks). Small
   entries get added as one line to the log.
3. **Tests.** AI-drafted tests are allowed but we run them ourselves and inspect
   what they assert — a test that asserts the wrong thing is worse than no test.
4. **Results reporting.** Charts, result tables, and copy for `report.md` may be
   AI-drafted from numbers **we** computed; AI is not allowed to invent or round
   numbers. We verify every figure against the actual output files.
5. **Documentation** (like this file). AI-drafted docs go in only if they match
   what the code actually does; we review before committing.

## Review checklist before each commit

- [ ] Both teammates understand every changed line.
- [ ] This log has a dated entry for any new AI use.
- [ ] Numbers in `report.md` match the actual outputs (no AI-invented results).
- [ ] No real bank data is staged for commit.
- [ ] Instructor policy is still met; if unsure, we ask first.