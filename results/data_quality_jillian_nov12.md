# Data quality checks - jillian

Source: `jillian_transactions_labeled.csv` (built by `ml/bank_statment_processor.py`)

## Source files

```
                                                bank_folder  raw_rows  first_date   last_date downloaded_(file_mtime)
file                                                                                                                 
bofa_credit_card.csv                        bank of america       393  2024-12-05  2025-12-03              2026-11-03
jillian_bofa_checking_2025_transcribed.csv           manual       275  2024-12-30  2025-12-29              2026-11-12
```

## Cleaning steps

```
                                                                   rows
step                                                                   
raw rows loaded                                                     668
dropped by noise/transfer filter (and approved duplicate removal)    88
rows after cleaning                                                 580
noise-pattern rows remaining (should be 0)                            0
```

## Duplicates and missing values

- Possible duplicates (same date, raw description, amount, bank, account): **1** (could be real repeat purchases; the processor flags these for review)

```
                 missing
date                   0
raw_description        0
type                   0
amount                 0
main_category          0
sub_category           0
bank                   0
account                0
```

## Coverage

```
                             rows  first_date   last_date
bank            account                                  
bank of america checking      187  2024-12-30  2025-12-29
                credit card   393  2024-12-05  2025-12-03
```

- Longest gap between transactions: **8 days**
- Months with no transactions: none

```
         transactions
date                 
2024-12            11
2025-01            39
2025-02            47
2025-03            61
2025-04            44
2025-05            60
2025-06            53
2025-07            47
2025-08            24
2025-09            63
2025-10            73
2025-11            37
2025-12            21
```

## Labels

- Labeled rows: **580 / 580**

```
                     transactions  share
main_category                           
Food & Dining                 310  0.534
Education                      64  0.110
Income                         56  0.097
Transportation                 44  0.076
Shopping & Supplies            31  0.053
Transfer                       27  0.047
Entertainment                  20  0.034
Health & Wellness              15  0.026
Miscellaneous                   7  0.012
Travel & Lodging                6  0.010
```

- Distinct sub-categories used: 26
- Distinct descriptions: 335
