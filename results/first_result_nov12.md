# First result: baseline, TF-IDF KNN, and embedding KNN (main_category)

Models: `most_common` (always the most frequent training label), `knn_tfidf` (character TF-IDF + standardized amount multiplied by 25 + 5-NN, cosine), and `knn_embeddings` (all-MiniLM-L6-v2 sentence embeddings + standardized amount multiplied by 25 + 5-NN, cosine). Rerun: `uv run python -m ml.first_result --jillian <csv> --savanna <csv>`

### Jillian: within-person (GroupKFold, 5 folds, grouped by description)

580 labeled rows, 335 distinct descriptions, 10 main categories.

```
                accuracy_mean  accuracy_sd  macro_f1_mean  macro_f1_sd
model                                                                 
most_common             0.534        0.118          0.074        0.015
knn_tfidf               0.574        0.159          0.442        0.059
knn_embeddings          0.672        0.170          0.488        0.061
```

Embedding KNN per-category report (pooled out-of-fold predictions):

```
                     precision  recall  f1-score  support
Education                0.598   0.766     0.671       64
Entertainment            0.200   0.200     0.200       20
Food & Dining            0.822   0.745     0.782      310
Health & Wellness        0.846   0.733     0.786       15
Income                   0.957   0.804     0.874       56
Miscellaneous            0.000   0.000     0.000        7
Shopping & Supplies      0.217   0.161     0.185       31
Transfer                 0.800   0.889     0.842       27
Transportation           0.232   0.432     0.302       44
Travel & Lodging         1.000   0.333     0.500        6
```

Embedding KNN confusion matrix (rows = true category, columns = predicted; bold diagonal = correct, · = 0):

| true ↓ / predicted → | Edu | Ent | Food | Hlth | Inc | Misc | Shop | Xfer | Trans | Trvl | total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Education (Edu) | **49** | 7 | 6 | 2 | · | · | · | · | · | · | 64 |
| Entertainment (Ent) | 7 | **4** | 6 | · | · | · | 1 | · | 2 | · | 20 |
| Food & Dining (Food) | 10 | 4 | **231** | · | · | · | 11 | · | 54 | · | 310 |
| Health & Wellness (Hlth) | 2 | · | 2 | **11** | · | · | · | · | · | · | 15 |
| Income (Inc) | · | · | 5 | · | **45** | · | · | 6 | · | · | 56 |
| Miscellaneous (Misc) | 1 | · | 4 | · | · | **·** | 1 | · | 1 | · | 7 |
| Shopping & Supplies (Shop) | 8 | 2 | 10 | · | · | · | **5** | · | 6 | · | 31 |
| Transfer (Xfer) | · | · | 1 | · | 2 | · | · | **24** | · | · | 27 |
| Transportation (Trans) | 4 | 2 | 15 | · | · | · | 4 | · | **19** | · | 44 |
| Travel & Lodging (Trvl) | 1 | 1 | 1 | · | · | · | 1 | · | · | **2** | 6 |

### Savanna: within-person (GroupKFold, 5 folds, grouped by description)

998 labeled rows, 421 distinct descriptions, 13 main categories.

```
                accuracy_mean  accuracy_sd  macro_f1_mean  macro_f1_sd
model                                                                 
most_common             0.348        0.105          0.042        0.009
knn_tfidf               0.504        0.125          0.440        0.068
knn_embeddings          0.558        0.160          0.457        0.066
```

Embedding KNN per-category report (pooled out-of-fold predictions):

```
                       precision  recall  f1-score  support
Education                  0.333   0.500     0.400       14
Entertainment              0.741   0.635     0.684       63
Food & Dining              0.625   0.758     0.685      347
Gifts & Donations          0.344   0.786     0.478       14
Health & Wellness          0.423   0.282     0.338       39
Housing & Bills            0.038   0.020     0.027       49
Income                     0.646   0.988     0.781       83
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
| Food & Dining (Food) | 2 | · | **263** | · | 5 | · | 13 | 1 | · | 57 | · | 4 | 2 | 347 |
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
most_common        0.534     0.070     580
knn_tfidf          0.403     0.177     580
knn_embeddings     0.531     0.193     580
```

### Cross-person: train Jillian -> test Savanna

0.0% of Savanna's rows have a description that also appears in Jillian's data.

```
                accuracy  macro_f1  n_test
model                                     
most_common        0.348     0.040     998
knn_tfidf          0.184     0.089     998
knn_embeddings     0.307     0.105     998
```
