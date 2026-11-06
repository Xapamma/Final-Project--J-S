# First result: baseline vs. KNN (main_category)

Models: `most_common` (always the most frequent training label), `knn_tfidf` (Term Frequency-Inverse Document Frequency + 5-NN, cosine). Rerun: `uv run python -m ml.first_result --jillian <csv> --savanna <csv>`

### Jillian: within-person (GroupKFold, 5 folds, grouped by description)

393 labeled rows, 166 distinct descriptions, 10 main categories.

```
             accuracy_mean  accuracy_sd  macro_f1_mean  macro_f1_sd
model                                                              
most_common          0.624        0.215          0.101        0.031
knn_tfidf            0.672        0.217          0.361        0.053
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

KNN confusion matrix (rows = true category, columns = predicted; bold diagonal = correct, · = 0):

| true ↓ / predicted → | Edu | Ent | Food | Hlth | Inc | Misc | Shop | Xfer | Trans | Trvl | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Education (Edu) | **39** | · | 3 | · | · | · | · | 4 | · | 1 | 47 |
| Entertainment (Ent) | 2 | **1** | 12 | · | · | · | · | · | 1 | 1 | 17 |
| Food & Dining (Food) | 1 | 2 | **197** | · | · | · | · | · | 45 | · | 245 |
| Health & Wellness (Hlth) | · | · | 3 | **·** | · | · | · | · | · | · | 3 |
| Income (Inc) | 1 | 1 | · | · | **·** | · | · | · | · | · | 2 |
| Miscellaneous (Misc) | · | 1 | 3 | · | · | **·** | 1 | · | · | · | 5 |
| Shopping & Supplies (Shop) | · | 1 | 18 | · | · | · | **4** | · | 1 | · | 24 |
| Transfer (Xfer) | · | · | · | · | · | · | · | **14** | · | · | 14 |
| Transportation (Trans) | · | 2 | 24 | · | · | · | · | 1 | **4** | · | 31 |
| Travel & Lodging (Trvl) | · | · | · | · | · | · | · | · | · | **5** | 5 |

### Savanna: within-person (GroupKFold, 5 folds, grouped by description)

998 labeled rows, 421 distinct descriptions, 13 main categories.

```
             accuracy_mean  accuracy_sd  macro_f1_mean  macro_f1_sd
model                                                              
most_common          0.348        0.105          0.042        0.009
knn_tfidf            0.599        0.148          0.487        0.078
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

KNN confusion matrix (rows = true category, columns = predicted; bold diagonal = correct, · = 0):

| true ↓ / predicted → | Edu | Ent | Food | Gift | Hlth | Hous | Inc | Misc | Sav | Shop | Xfer | Trans | Trvl | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Education (Edu) | **2** | 2 | 1 | · | 3 | 1 | 3 | · | · | 2 | · | · | · | 14 |
| Entertainment (Ent) | · | **46** | 7 | · | · | · | 2 | 6 | · | 1 | · | 1 | · | 63 |
| Food & Dining (Food) | 4 | 4 | **295** | 2 | 5 | 3 | 5 | 3 | 1 | 8 | · | 15 | 2 | 347 |
| Gifts & Donations (Gift) | · | · | 1 | **13** | · | · | · | · | · | · | · | · | · | 14 |
| Health & Wellness (Hlth) | 8 | 1 | 11 | · | **14** | · | · | 2 | · | 2 | 1 | · | · | 39 |
| Housing & Bills (Hous) | 5 | · | · | · | · | **1** | · | 12 | · | · | 19 | 12 | · | 49 |
| Income (Inc) | 3 | · | · | · | 1 | 1 | **75** | 2 | · | 1 | · | · | · | 83 |
| Miscellaneous (Misc) | 1 | · | 11 | 1 | · | 3 | · | **18** | 1 | 3 | · | · | 1 | 39 |
| Savings & Investments (Sav) | · | · | · | · | · | · | · | 1 | **47** | · | · | 83 | · | 131 |
| Shopping & Supplies (Shop) | 2 | 3 | 24 | · | 2 | · | 6 | 2 | · | **35** | · | 2 | · | 76 |
| Transfer (Xfer) | · | · | · | · | · | 15 | 1 | · | · | · | **2** | · | · | 18 |
| Transportation (Trans) | 1 | 1 | 42 | · | 2 | 1 | · | · | 1 | 2 | · | **47** | · | 97 |
| Travel & Lodging (Trvl) | · | · | 19 | · | · | · | · | 3 | · | 3 | · | · | **3** | 28 |

### Cross-person: train Savanna -> test Jillian

0.0% of Jillian's rows have a description that also appears in Savanna's data.

```
             accuracy  macro_f1  n_test
model                                  
most_common     0.623     0.077     393
knn_tfidf       0.578     0.270     393
```

### Cross-person: train Jillian -> test Savanna

0.0% of Savanna's rows have a description that also appears in Jillian's data.

```
             accuracy  macro_f1  n_test
model                                  
most_common     0.348     0.040     998
knn_tfidf       0.376     0.118     998
```
