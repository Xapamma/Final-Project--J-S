# First result: baselines vs. KNN (main_category)

Models: `most_common` (always the most frequent training label), `merchant_lookup` (label seen most often for that exact description; falls back to most_common), `knn_tfidf` (character TF-IDF + 5-NN, cosine). Rerun: `uv run python -m ml.first_result --jillian <csv> --savanna <csv>`

### Jillian: within-person (GroupKFold, 5 folds, grouped by description)

393 labeled rows, 166 distinct descriptions, 10 main categories.

```
                 accuracy_mean  accuracy_sd  macro_f1_mean  macro_f1_sd
model                                                                  
most_common              0.624        0.215          0.101        0.031
merchant_lookup          0.624        0.215          0.101        0.031
knn_tfidf                0.672        0.217          0.361        0.053
```

KNN per-category report (pooled out-of-fold predictions):

```
                     precision  recall  f1-score  support
Education                0.907   0.830     0.867       47
Entertainment            0.125   0.059     0.080       17
Food & Dining            0.758   0.804     0.780      245
Health & Wellness        0.000   0.000     0.000        3
Income                   0.000   0.000     0.000        2
Miscellaneous            0.000   0.000     0.000        5
Shopping & Supplies      0.800   0.167     0.276       24
Transfer                 0.737   1.000     0.848       14
Transportation           0.078   0.129     0.098       31
Travel & Lodging         0.714   1.000     0.833        5
```

KNN confusion matrix (rows = true, columns = predicted):

```
predicted            Education  Entertainment  Food & Dining  Health & Wellness  Income  Miscellaneous  Shopping & Supplies  Transfer  Transportation  Travel & Lodging
true                                                                                                                                                                   
Education                   39              0              3                  0       0              0                    0         4               0                 1
Entertainment                2              1             12                  0       0              0                    0         0               1                 1
Food & Dining                1              2            197                  0       0              0                    0         0              45                 0
Health & Wellness            0              0              3                  0       0              0                    0         0               0                 0
Income                       1              1              0                  0       0              0                    0         0               0                 0
Miscellaneous                0              1              3                  0       0              0                    1         0               0                 0
Shopping & Supplies          0              1             18                  0       0              0                    4         0               1                 0
Transfer                     0              0              0                  0       0              0                    0        14               0                 0
Transportation               0              2             24                  0       0              0                    0         1               4                 0
Travel & Lodging             0              0              0                  0       0              0                    0         0               0                 5
```

### Savanna: within-person (GroupKFold, 5 folds, grouped by description)

998 labeled rows, 421 distinct descriptions, 13 main categories.

```
                 accuracy_mean  accuracy_sd  macro_f1_mean  macro_f1_sd
model                                                                  
most_common              0.348        0.105          0.042        0.009
merchant_lookup          0.348        0.105          0.042        0.009
knn_tfidf                0.599        0.148          0.487        0.078
```

KNN per-category report (pooled out-of-fold predictions):

```
                       precision  recall  f1-score  support
Education                  0.077   0.143     0.100       14
Entertainment              0.807   0.730     0.767       63
Food & Dining              0.718   0.850     0.778      347
Gifts & Donations          0.812   0.929     0.867       14
Health & Wellness          0.519   0.359     0.424       39
Housing & Bills            0.040   0.020     0.027       49
Income                     0.815   0.904     0.857       83
Miscellaneous              0.367   0.462     0.409       39
Savings & Investments      0.940   0.359     0.519      131
Shopping & Supplies        0.614   0.461     0.526       76
Transfer                   0.091   0.111     0.100       18
Transportation             0.294   0.485     0.366       97
Travel & Lodging           0.500   0.107     0.176       28
```

KNN confusion matrix (rows = true, columns = predicted):

```
predicted              Education  Entertainment  Food & Dining  Gifts & Donations  Health & Wellness  Housing & Bills  Income  Miscellaneous  Savings & Investments  Shopping & Supplies  Transfer  Transportation  Travel & Lodging
true                                                                                                                                                                                                                                
Education                      2              2              1                  0                  3                1       3              0                      0                    2         0               0                 0
Entertainment                  0             46              7                  0                  0                0       2              6                      0                    1         0               1                 0
Food & Dining                  4              4            295                  2                  5                3       5              3                      1                    8         0              15                 2
Gifts & Donations              0              0              1                 13                  0                0       0              0                      0                    0         0               0                 0
Health & Wellness              8              1             11                  0                 14                0       0              2                      0                    2         1               0                 0
Housing & Bills                5              0              0                  0                  0                1       0             12                      0                    0        19              12                 0
Income                         3              0              0                  0                  1                1      75              2                      0                    1         0               0                 0
Miscellaneous                  1              0             11                  1                  0                3       0             18                      1                    3         0               0                 1
Savings & Investments          0              0              0                  0                  0                0       0              1                     47                    0         0              83                 0
Shopping & Supplies            2              3             24                  0                  2                0       6              2                      0                   35         0               2                 0
Transfer                       0              0              0                  0                  0               15       1              0                      0                    0         2               0                 0
Transportation                 1              1             42                  0                  2                1       0              0                      1                    2         0              47                 0
Travel & Lodging               0              0             19                  0                  0                0       0              3                      0                    3         0               0                 3
```

### Cross-person: train Savanna -> test Jillian

0.0% of Jillian's rows have a description that also appears in Savanna's data.

```
                 accuracy  macro_f1  n_test
model                                      
most_common         0.623     0.077     393
merchant_lookup     0.623     0.077     393
knn_tfidf           0.578     0.270     393
```

### Cross-person: train Jillian -> test Savanna

0.0% of Savanna's rows have a description that also appears in Jillian's data.

```
                 accuracy  macro_f1  n_test
model                                      
most_common         0.348     0.040     998
merchant_lookup     0.348     0.040     998
knn_tfidf           0.376     0.118     998
```
