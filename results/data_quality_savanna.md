# Data quality checks - savanna

Source: `categorized_transactions_reviewed(categorized_transactions_review).csv` (built by the Streamlit app export (no `raw_description`, so cleaning-step counts are unavailable))

## Source files

(raw files not available on this machine)

## Cleaning steps

Not available: needs the raw bank files and the processor's `raw_description` column.

## Duplicates and missing values

- Possible duplicates (same date, description, amount, bank, account): **77** (could be real repeat purchases; the processor flags these for review)

```
               missing
date                 0
description          0
type                 0
amount               0
main_category        0
sub_category         0
bank                 0
account              0
```

## Coverage

```
                          rows  first_date   last_date
bank        account                                   
capital one credit card    263  2025-07-10  2025-12-30
goldenwest  checking       273  2025-01-03  2025-12-31
            gold account    15  2025-01-31  2025-12-31
            money market    17  2025-03-31  2025-12-31
sofi        checking        39  2025-01-02  2025-03-26
            savings         24  2025-01-07  2025-12-31
us bank     checking       367  2025-01-02  2025-12-29
```

- Longest gap between transactions: **4 days**
- Months with no transactions: none

```
         transactions
date                 
2025-01            69
2025-02            76
2025-03            88
2025-04           144
2025-05            64
2025-06            55
2025-07            99
2025-08            76
2025-09            66
2025-10            75
2025-11            92
2025-12            94
```

## Labels

- Labeled rows: **998 / 998**

```
                       transactions  share
main_category                             
Food & Dining                   347  0.348
Savings & Investments           131  0.131
Transportation                   97  0.097
Income                           83  0.083
Shopping & Supplies              76  0.076
Entertainment                    63  0.063
Housing & Bills                  49  0.049
Health & Wellness                39  0.039
Miscellaneous                    39  0.039
Travel & Lodging                 28  0.028
Transfer                         18  0.018
Education                        14  0.014
Gifts & Donations                14  0.014
```

- Distinct sub-categories used: 44
- Distinct original descriptions: 423
