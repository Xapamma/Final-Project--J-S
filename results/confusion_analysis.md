# Working investigation: confusion pairs and TF-IDF vs embeddings

Rerun: `uv run python -m ml.confusion_analysis --jillian <csv> --savanna <csv>`. `exact` groups GroupKFold folds by the exact description (as in first_result.md); `normalized` groups by the description with digits and punctuation removed, so rows that differ only by an ID, date, or confirmation number stay in the same fold.

## Jillian

580 labeled rows; 335 exact descriptions vs 233 normalized descriptions.

Mean of 5 GroupKFold folds:

```
                accuracy_exact  accuracy_normalized  macro_f1_exact  macro_f1_normalized
model                                                                                   
most_common              0.534                0.534           0.074                0.080
knn_tfidf                0.574                0.481           0.442                0.329
knn_embeddings           0.672                0.586           0.488                0.385
```

Per-category F1 (normalized grouping, pooled out-of-fold):

```
                     support  f1_tfidf  f1_embeddings      better
Food & Dining            310     0.663          0.762  embeddings
Education                 64     0.338          0.587  embeddings
Income                    56     0.587          0.577         tie
Transportation            44     0.349          0.343         tie
Shopping & Supplies       31     0.000          0.145  embeddings
Transfer                  27     0.535          0.438       tfidf
Entertainment             20     0.074          0.211  embeddings
Health & Wellness         15     0.000          0.000         tie
Miscellaneous              7     0.000          0.000         tie
Travel & Lodging           6     0.500          0.500         tie
```

Biggest confusion pairs (row counts, normalized grouping):

| true | predicted | TF-IDF | embeddings | top merchants (embeddings errors) |
|---|---|---:|---:|---|
| Food & Dining | Transportation | 59 | 54 | CHEVRON UT (46), CAFE RIO UT (1), SHELL OIL KENNEWICK (1), CHEVRON MOUNTAIN HOMEID (1) |
| Food & Dining | Education | 72 | 17 | VENMO (8), UNIVERSITY (5), RICH'S BURGERS N (1), WM SUPERCENTER TWIN (1) |
| Income | Transfer | 24 | 23 | UNIVERSITY (14), ONLINE BANKING TRANSFER (5), MAG USA IL (1), IRS TREAS (1) |
| Shopping & Supplies | Education | 26 | 8 | LULULEMON CHANDLER CHANDLER (1), LS SOEL BOUTIQUE (1), CALLED SURF UT (1), NORDSTROM SCOTTSDALE AZ (1) |
| Health & Wellness | Education | 14 | 13 | GYM (12), FAIRBANKS ORTHODONTICS UT (1) |
| Education | Entertainment | 15 | 8 | UNIVERSITY (7), AP SCORE REPORT (1) |
| Entertainment | Education | 15 | 8 | UNIVERSITY (1), SOLITUDE MOUNTAIN RESO (1), VIVID SEATS THE (1), BRIGHTON RESORT UT (1) |
| Transfer | Income | 8 | 13 | VENMO (11), ONLINE PAYMENT FROM (2) |

## Savanna

998 labeled rows; 421 exact descriptions vs 341 normalized descriptions.

Mean of 5 GroupKFold folds:

```
                accuracy_exact  accuracy_normalized  macro_f1_exact  macro_f1_normalized
model                                                                                   
most_common              0.348                0.348           0.042                0.045
knn_tfidf                0.504                0.527           0.440                0.330
knn_embeddings           0.558                0.596           0.457                0.330
```

Per-category F1 (normalized grouping, pooled out-of-fold):

```
                       support  f1_tfidf  f1_embeddings      better
Food & Dining              347     0.728          0.764  embeddings
Savings & Investments      131     0.948          0.945         tie
Transportation              97     0.402          0.494  embeddings
Income                      83     0.664          0.827  embeddings
Shopping & Supplies         76     0.275          0.230       tfidf
Entertainment               63     0.510          0.724  embeddings
Housing & Bills             49     0.013          0.071  embeddings
Health & Wellness           39     0.289          0.081       tfidf
Miscellaneous               39     0.076          0.102  embeddings
Travel & Lodging            28     0.143          0.097       tfidf
Transfer                    18     0.000          0.000         tie
Education                   14     0.000          0.194  embeddings
Gifts & Donations           14     0.242          0.045       tfidf
```

Biggest confusion pairs (row counts, normalized grouping):

| true | predicted | TF-IDF | embeddings | top merchants (embeddings errors) |
|---|---|---:|---:|---|
| Transportation | Food & Dining | 24 | 39 | MAVERIK (14), RECURRING DEBIT PURCHASE (12), COSTCO GAS (3), JACKSONS FOOD S (1) |
| Food & Dining | Income | 39 | 13 | UTAH CAREER CENTER (2), COSTCO (1), JEEVAS GREEK CAFE (1), CHICK FIL A (1) |
| Shopping & Supplies | Food & Dining | 7 | 31 | ROSS STORES (5), HOBBY LOBBY (5), THE HOME DEPOT (4), DEBIT PURCHASE VISA (1) |
| Food & Dining | Shopping & Supplies | 5 | 29 | HOME DEPOT RIVERTON (17), DEBIT PURCHASE VISA (3), LEATHERBY'S FAMILY CREAM (2), IN N OUT (1) |
| Food & Dining | Housing & Bills | 26 | 3 | WM SUPERCENTER (2), COSTCO WHSE (1) |
| Housing & Bills | Travel & Lodging | 12 | 12 | WEB AUTHORIZED PMT (12) |
| Housing & Bills | Transfer | 12 | 12 | WEB AUTHORIZED PMT (12) |
| Shopping & Supplies | Housing & Bills | 21 | 3 | AMAZON MKTPLACE PMTS (1), RC WILLEY HOME (1), LAGOON RFM (1) |

