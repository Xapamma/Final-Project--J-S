# Data quality checks - jillian

Source: `jillian_transactions_labeled.csv` (built by `ml/bank_statment_processor.py`)

## Source files

```
                          bank_folder  raw_rows  first_date   last_date downloaded_(file_mtime)
file                                                                                           
bofa_credit_card.csv  bank of america       393  2024-12-05  2025-12-03              2026-11-03
```

## Cleaning steps

```
                                            rows
step                                            
raw rows loaded                              393
dropped by noise/transfer filter               0
rows after processing                        393
noise-pattern rows remaining (should be 0)     0
```

## Duplicates and missing values

- Possible duplicates (same date, description, amount, bank, account): **1** (not removed automatically - could be real repeat purchases)

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
bank            account                                  
bank of america credit card   393  2024-12-05  2025-12-03
```

- Longest gap between transactions: **8 days**
- Months with no transactions: none

```
         transactions
date                 
2024-12            10
2025-01            26
2025-02            29
2025-03            45
2025-04            29
2025-05            45
2025-06            27
2025-07            28
2025-08            18
2025-09            55
2025-10            51
2025-11            28
2025-12             2
```

## Labels

- Labeled rows: **393 / 393**

```
                     transactions  share
main_category                           
Food & Dining                 245  0.623
Education                      47  0.120
Transportation                 31  0.079
Shopping & Supplies            24  0.061
Entertainment                  17  0.043
Transfer                       14  0.036
Miscellaneous                   5  0.013
Travel & Lodging                5  0.013
Health & Wellness               3  0.008
Income                          2  0.005
```

- Distinct sub-categories used: 21
- Distinct descriptions: 166
