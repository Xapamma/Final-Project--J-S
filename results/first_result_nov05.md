# First result: baseline, TF-IDF KNN, and embedding KNN (main_category)

Models: `most_common` (always the most frequent training label), `knn_tfidf` (character TF-IDF + standardized amount multiplied by 25 + 5-NN, cosine), and `knn_embeddings` (all-MiniLM-L6-v2 sentence embeddings + standardized amount multiplied by 25 + 5-NN, cosine). Rerun: `uv run python -m ml.first_result --jillian <csv> --savanna <csv>`

### Jillian: within-person (GroupKFold, 5 folds, grouped by description)

393 labeled rows, 166 distinct descriptions, 10 main categories.

```
                accuracy_mean  accuracy_sd  macro_f1_mean  macro_f1_sd
model                                                                 
most_common             0.624        0.215          0.101        0.031
knn_tfidf               0.483        0.155          0.315        0.158
knn_embeddings          0.603        0.199          0.314        0.113
```

Embedding KNN per-category report (pooled out-of-fold predictions):

```
                     precision  recall  f1-score  support
Education                0.200   0.021     0.038       47
Entertainment            0.150   0.176     0.162       17
Food & Dining            0.745   0.788     0.766      245
Health & Wellness        0.000   0.000     0.000        3
Income                   0.000   0.000     0.000        2
Miscellaneous            0.000   0.000     0.000        5
Shopping & Supplies      0.125   0.167     0.143       24
Transfer                 0.824   1.000     0.903       14
Transportation           0.370   0.645     0.471       31
Travel & Lodging         0.333   0.400     0.364        5
```

Embedding KNN confusion matrix (rows = true category, columns = predicted; bold diagonal = correct, · = 0):

| true ↓ / predicted → | Edu | Ent | Food | Hlth | Inc | Misc | Shop | Xfer | Trans | Trvl | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Education (Edu) | **1** | 1 | 44 | · | · | · | 1 | · | · | · | 47 |
| Entertainment (Ent) | · | **3** | 5 | · | · | · | 7 | · | 2 | · | 17 |
| Food & Dining (Food) | 4 | 2 | **193** | · | · | · | 19 | · | 24 | 3 | 245 |
| Health & Wellness (Hlth) | · | 1 | 1 | **·** | · | · | · | · | 1 | · | 3 |
| Income (Inc) | · | · | · | · | **·** | · | · | 2 | · | · | 2 |
| Miscellaneous (Misc) | · | 1 | 2 | · | · | **·** | · | 1 | 1 | · | 5 |
| Shopping & Supplies (Shop) | · | 5 | 10 | · | · | · | **4** | · | 5 | · | 24 |
| Transfer (Xfer) | · | · | · | · | · | · | · | **14** | · | · | 14 |
| Transportation (Trans) | · | 5 | 4 | · | · | · | 1 | · | **20** | 1 | 31 |
| Travel & Lodging (Trvl) | · | 2 | · | · | · | · | · | · | 1 | **2** | 5 |

### Savanna: within-person (GroupKFold, 5 folds, grouped by description)

998 labeled rows, 421 distinct descriptions, 13 main categories.

```
                accuracy_mean  accuracy_sd  macro_f1_mean  macro_f1_sd
model                                                                 
most_common             0.348        0.105          0.042        0.009
knn_tfidf               0.505        0.128          0.440        0.072
knn_embeddings          0.559        0.160          0.457        0.066
```

Embedding KNN per-category report (pooled out-of-fold predictions):

```
                       precision  recall  f1-score  support
Education                  0.333   0.500     0.400       14
Entertainment              0.741   0.635     0.684       63
Food & Dining              0.626   0.761     0.687      347
Gifts & Donations          0.344   0.786     0.478       14
Health & Wellness          0.423   0.282     0.338       39
Housing & Bills            0.038   0.020     0.027       49
Income                     0.651   0.988     0.785       83
Miscellaneous              0.360   0.231     0.281       39
Savings & Investments      1.000   0.267     0.422      131
Shopping & Supplies        0.198   0.250     0.221       76
Transfer                   0.143   0.111     0.125       18
Transportation             0.763   0.763     0.763       97
Travel & Lodging           0.125   0.107     0.115       28
```

Embedding KNN confusion matrix (rows = true category, columns = predicted; bold diagonal = correct, · = 0):

| true ↓ / predicted → | Edu | Ent | Food | Gift | Hlth | Hous | Inc | Misc | Sav | Shop | Xfer | Trans | Trvl | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Education (Edu) | **7** | 2 | 1 | · | 1 | 2 | · | 1 | · | · | · | · | · | 14 |
| Entertainment (Ent) | 2 | **40** | 7 | · | 1 | · | 3 | 3 | · | 3 | · | 2 | 2 | 63 |
| Food & Dining (Food) | 2 | · | **264** | · | 5 | · | 12 | 1 | · | 57 | · | 4 | 2 | 347 |
| Gifts & Donations (Gift) | 1 | · | · | **11** | · | · | · | 1 | · | · | · | · | 1 | 14 |
| Health & Wellness (Hlth) | 1 | 7 | 7 | 9 | **11** | · | · | 2 | · | 1 | · | 1 | · | 39 |
| Housing & Bills (Hous) | · | · | 7 | 7 | · | **1** | · | 5 | · | 5 | 12 | · | 12 | 49 |
| Income (Inc) | · | · | 1 | · | · | · | **82** | · | · | · | · | · | · | 83 |
| Miscellaneous (Misc) | 3 | 1 | 7 | 3 | 4 | 1 | 7 | **9** | · | 4 | · | · | · | 39 |
| Savings & Investments (Sav) | · | · | 79 | · | 1 | 2 | 12 | · | **35** | 1 | · | 1 | · | 131 |
| Shopping & Supplies (Shop) | 1 | 2 | 34 | 1 | 2 | 2 | 7 | 1 | · | **19** | · | 5 | 2 | 76 |
| Transfer (Xfer) | · | · | · | · | · | 11 | 1 | · | · | · | **2** | 4 | · | 18 |
| Transportation (Trans) | 4 | 1 | 9 | · | 1 | 1 | 2 | 2 | · | 1 | · | **74** | 2 | 97 |
| Travel & Lodging (Trvl) | · | 1 | 6 | 1 | · | 6 | · | · | · | 5 | · | 6 | **3** | 28 |

### Cross-person: train Savanna -> test Jillian

0.0% of Jillian's rows have a description that also appears in Savanna's data.

```
                accuracy  macro_f1  n_test
model                                     
most_common        0.623     0.077     393
knn_tfidf          0.384     0.151     393
knn_embeddings     0.496     0.135     393
```

### Cross-person: train Jillian -> test Savanna

0.0% of Savanna's rows have a description that also appears in Jillian's data.

```
                accuracy  macro_f1  n_test
model                                     
most_common        0.348     0.040     998
knn_tfidf          0.185     0.064     998
knn_embeddings     0.274     0.089     998
```
