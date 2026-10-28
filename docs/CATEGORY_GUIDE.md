# Category Guide

STAT 486 Fall 2026 – Savanna & Jillian
Updated: October 27, 2026

This is the **labeling manual** we promised in the project plan ("We will write
the category guide together"). It defines the category vocabulary and the rules
we both use when labeling so our ML training data is consistent.

This guide is **not** the old rule-based categorizer. It only says how *humans*
label; nothing in the code assigns these automatically.

## 1. How the vocabulary works

Every transaction gets a **main category** and a **sub-category** drawn from
this list (the same list powers the dropdowns in the Streamlit app):

| Main category | Sub-categories |
|---|---|
| Income | Paychecks, Refunds, CC Rewards, Other Income, Savings / Other Withdrawals |
| Transfer | N/A |
| Food & Dining | Groceries, Restaurants/Dining, Fast Food, Snacks |
| Transportation | Vehicle Payments, Auto Insurance, Fuel, Licensing, Parking, Repairs, Maintenance |
| Health & Wellness | Doctor/Dentist, Medicine/Drugs, Personal Care, Health Insurance |
| Travel & Lodging | Airfare, Lodging, Travel & Commute |
| Shopping & Supplies | General Retail, Home Improvement, Hobbies, Clothing |
| Housing & Bills | Mortgage/Rent, Phone, Cable/Internet, Service Fees, Utilities, Maintenance/Repairs |
| Gifts & Donations | Tithing, Fast Offerings, Hum Aid, Gift |
| Education | Education, Tutition, Books |
| Entertainment | Gaming, Subscriptions, Recreation, Tickets |
| Savings & Investments | Dividends, Investments, Emergency Fund, Retirement, House Down Payment |
| Miscellaneous | Fees & Charges, Other Services |
| Vacations / Travel | Airfare, Travel, Lodging, Souvenirs |

## 2. Labeling rules (follow these every time)

1. **Label what the money was for**, not where it appeared. "Amazon" is a
   merchant, not a category — decide groceries vs. clothing vs. books by the
   item.
2. **Positive amounts = money in.** Assign `Income` (e.g. Paychecks, Refunds,
   CC Rewards) unless the credit is a refund of something we already spent.
3. **Internal transfers → `Transfer / N/A`.** Money moved between our own
   accounts is not income or spending. If the importer missed one, set the main
   category to `Transfer`; the app sets sub-category `N/A` and moves the row to
   the excluded-review list.
4. **When in doubt, pick the closest specific sub-category** over a vague one.
   If truly none fit, use `Miscellaneous / Other Services`.
5. **Be consistent between us.** If we label the same merchant differently,
   agree on one and add it to the "shared sample notes" below. These
   disagreements are expected and are part of our analysis (see report plan),
   but we want the volume small.
6. **Never leave a label blank** in the final labeled dataset; blank means
   "not labeled yet," and the ML models need every row labeled.

## 3. Special cases

* `Salary/Regular Income ... Goldenwest` → `Income / Paychecks`
* Bank `APY/dividend` deposits → `Income / CC Rewards` or
  `Savings & Investments / Dividends` (use the bank's own category as a hint).
* Card payments (e.g. "Payment Thank You") → usually internal transfers
  (`Transfer`), are auto-excluded for review, and can be restored if wrong.
* `.robinhood` deposits or brokerage names → `Savings & Investments / Investments`.

*(Add merchant-specific decisions here as we label.)*

## 4. Shared sample notes

Per the plan we each label a shared set of transactions and compare. Record
examples where we disagreed and what we decided:

| Description | Person A label | Person B label | Agreed label |
|---|---|---|---|
| _pending_ | _pending_ | _pending_ | _pending_ |

## 5. Why this matters for ML

The models in `ml/ml_models.py` learn from `description` (and later
`raw_description`, amount, bank, etc.) to predict these labels. Garbage-in /
garbage-out: inconsistent labels cap how good any model can be — that is exactly
the effect our project wants to measure ("labeling differences between us").