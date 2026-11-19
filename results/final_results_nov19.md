# Final Results: What Information Helps Categorize Transactions?

**Date:** November 19, 2026

## Original project questions

1. After cleaning and combining our personal bank statements into one dataset, which machine learning techniques most accurately categorize transactions into main categories and subcategories?
2. Can a model trained on one person's bank statements correctly categorize the statements of another?

### Questions guiding this report

To investigate the original questions, we focus on three comparisons:

1. Does adding amount, transaction type, or account to the description improve predictions within each person's data?
2. Do those features help when a model is trained on one person and tested on the other?
3. Which model performs best, particularly across less common categories?

This report evaluates main categories using each person's dataset separately; it does not evaluate subcategories or train a model on a combined dataset. The [answers to the original questions](#answers-to-the-original-project-questions) explain what the results can and cannot tell us.

## How to read these results

These experiments predict main categories for 998 Savanna and 580 Jillian labeled transactions. Macro F1 gives each category equal weight; accuracy is the share of predictions that are correct. For within-person results, predictions are pooled from five outer folds grouped by description (lowercased, with surrounding whitespace removed), so identical descriptions never appear in both training and test data. For cross-person results, the whole labeled dataset of one person is used for training and the other's for testing. Jillian's dataset is 393 credit-card and 187 checking transactions, the same version checked in the [November 12 quality report](data_quality_jillian_nov12.md). Both datasets exclude internal transfers between a person's own accounts (for Jillian, 88 rows: "Keep the Change" round-ups, savings transfers, and credit-card payments from checking); credit-card payments seen from the other side remain labeled `Transfer` (27 Jillian rows, 18 Savanna rows).

The ablation uses fixed model settings and the same splits for each feature variant. The description-only column gives macro F1; the other columns show the change in macro F1 when that single feature is added. A positive change is better. TF-IDF KNN only tests amount; type and account are not tested for it. The baseline for each row is that same model with description alone. See the [complete aggregate and fold scores](tables/feature_ablation_scores.csv) for accuracy and exact values.

For the ablation, both KNN versions use five neighbors and cosine distance; the random forest uses 200 trees with no depth limit; and the neural network uses one 100-unit hidden layer, alpha 0.0001, a 0.001 learning rate, a 1,000-iteration limit, and early stopping. These settings are fixed, not retuned for each ablation variant.

## 1. Do extra features help within each person?

![Change in within-person macro F1 when adding amount](media/feature_amount_effect.svg)

Adding amount improves macro F1 in 7 of 8 person/model comparisons. The largest gain is +0.137 for Jillian's neural network; the only loss is -0.082 for Savanna's random forest. Type and account have model-dependent effects (account lowers macro F1 for three of Savanna's models); these comparisons do not show that every extra field helps every model.

### Savanna

| Model | Description F1 | + Amount | + Type | + Account |
| --- | --- | --- | --- | --- |
| TF-IDF KNN | 0.479 | +0.034 | — | — |
| Embedding KNN | 0.421 | +0.063 | +0.020 | -0.070 |
| Random forest | 0.499 | -0.082 | -0.052 | -0.009 |
| Neural network | 0.556 | +0.038 | +0.007 | -0.015 |

### Jillian

| Model | Description F1 | + Amount | + Type | + Account |
| --- | --- | --- | --- | --- |
| TF-IDF KNN | 0.575 | +0.011 | — | — |
| Embedding KNN | 0.524 | +0.010 | +0.014 | -0.025 |
| Random forest | 0.467 | +0.017 | +0.033 | +0.019 |
| Neural network | 0.505 | +0.137 | +0.050 | +0.055 |

## 2. Do extra features transfer across people?

The effect depends on the training direction. Adding amount improves all four models when training on Savanna and testing on Jillian (+0.015 to +0.060), but only one of four in the reverse direction (random forest, +0.072); it lowers TF-IDF KNN, embedding KNN, and the neural network when training on Jillian. Overall it improves 5 of 8 cross-person comparisons. These comparisons alone do not establish reliable transfer.

### Savanna to Jillian

| Model | Description F1 | + Amount | + Type | + Account |
| --- | --- | --- | --- | --- |
| TF-IDF KNN | 0.233 | +0.041 | — | — |
| Embedding KNN | 0.168 | +0.060 | +0.037 | +0.031 |
| Random forest | 0.142 | +0.026 | +0.014 | -0.020 |
| Neural network | 0.182 | +0.015 | +0.037 | +0.043 |

### Jillian to Savanna

| Model | Description F1 | + Amount | + Type | + Account |
| --- | --- | --- | --- | --- |
| TF-IDF KNN | 0.164 | -0.019 | — | — |
| Embedding KNN | 0.133 | -0.035 | +0.035 | -0.020 |
| Random forest | 0.071 | +0.072 | +0.065 | +0.001 |
| Neural network | 0.160 | -0.014 | +0.003 | +0.002 |

## 3. Which models performed best?

The table shows the highest observed macro F1 among the fixed-setting variants when training and testing within each person's data, and when training on one person and testing on the other. These are exploratory winners selected after looking at the results, not unbiased estimates for a model chosen in advance. “Savanna to Jillian” means training on Savanna and testing on Jillian; the reverse uses Jillian for training and Savanna for testing. The most-common-category baseline provides context.

| Evaluation | Model | Features | Accuracy | Macro F1 | Baseline accuracy | Baseline macro F1 |
| --- | --- | --- | --- | --- | --- | --- |
| within: Savanna | Neural network | description + amount | 0.652 | 0.594 | 0.348 | 0.040 |
| within: Jillian | Neural network | description + amount | 0.728 | 0.642 | 0.534 | 0.070 |
| cross: Savanna to Jillian | TF-IDF KNN | description + amount | 0.536 | 0.274 | 0.534 | 0.070 |
| cross: Jillian to Savanna | Embedding KNN | description + type | 0.440 | 0.167 | 0.348 | 0.040 |

The tuned-model comparison selects features and parameters using each outer training set before predicting its held-out rows. It evaluates the whole selection procedure, while the fixed-setting ablation isolates individual inputs. In the tuned within-person comparison, neural network had the highest accuracy and macro F1 for Savanna, and TF-IDF KNN had the highest accuracy and macro F1 for Jillian. Features and parameters can differ by outer fold.

| Model | Savanna accuracy | Savanna macro F1 | Jillian accuracy | Jillian macro F1 |
| --- | --- | --- | --- | --- |
| Most-common baseline | 0.348 | 0.040 | 0.534 | 0.070 |
| TF-IDF KNN | 0.601 | 0.504 | 0.745 | 0.623 |
| Embedding KNN | 0.566 | 0.459 | 0.660 | 0.537 |
| Random forest | 0.595 | 0.494 | 0.645 | 0.472 |
| Neural network | 0.657 | 0.542 | 0.666 | 0.538 |

In the tuned cross-person comparison, **no model beat the most-common baseline's accuracy when training on Savanna and testing on Jillian** (0.488–0.528 vs. 0.534); the models only beat it on macro F1 (best: TF-IDF KNN, 0.286 vs. 0.070). Training on Jillian and testing on Savanna, every model beat the baseline's accuracy of 0.348; embedding KNN was highest on both metrics (0.450 / 0.179).

| Train → test | Model | Selected features | Accuracy | Macro F1 |
| --- | --- | --- | --- | --- |
| Savanna → Jillian | Most-common baseline | — | 0.534 | 0.070 |
| Savanna → Jillian | TF-IDF KNN | description | 0.488 | 0.286 |
| Savanna → Jillian | Embedding KNN | description + type | 0.528 | 0.236 |
| Savanna → Jillian | Random forest | description + type | 0.521 | 0.156 |
| Savanna → Jillian | Neural network | description + amount | 0.512 | 0.219 |
| Jillian → Savanna | Most-common baseline | — | 0.348 | 0.040 |
| Jillian → Savanna | TF-IDF KNN | description | 0.378 | 0.170 |
| Jillian → Savanna | Embedding KNN | description + type | 0.450 | 0.179 |
| Jillian → Savanna | Random forest | description + amount + type | 0.436 | 0.140 |
| Jillian → Savanna | Neural network | description + amount | 0.376 | 0.146 |

![Within-person accuracy by model for Savanna and Jillian](media/accuracy_comparison.svg)

![Within-person macro F1 by model for Savanna and Jillian](media/macro_f1_comparison.svg)

The tuned comparison's [within-person scores](tables/tuned_comparison_nov19_scores.csv), [cross-person scores](tables/tuned_comparison_nov19_cross_person.csv), and [fold selections](tables/tuned_comparison_nov19_selections.csv) are also saved as tables.

## Evaluation and limitations

![Nested cross-validation workflow](media/cv_workflow.svg)

For each outer training set, feature selection and parameter tuning use the same inner training–validation splits in two rounds. First, validation scores compare feature sets using fixed model settings. Then the chosen features are kept while the same validation rows help compare parameter settings. Those tuning scores are not an independent evaluation of both choices; the held-out outer test fold evaluates the complete selection procedure.

During feature selection and parameter tuning, each inner fold fits data-dependent preprocessing—such as amount standardization and the TF-IDF vocabulary—using only that fold's inner training rows. It then oversamples rare categories in those same training rows. The inner validation rows are transformed using the training-fitted preprocessing and are not oversampled. The pretrained sentence-embedding model is used as-is and is not fitted to our transactions.

After selecting the features and parameters, the code fits preprocessing again using the **whole outer training set**, oversamples its rare categories, and refits the selected model. It applies that training-fitted preprocessing to the held-out outer test rows without oversampling them. Inner folds are stratified by category but do not group descriptions. When a category has only one row in an outer training set, feature selection and tuning fall back to the documented defaults.

The neural network also uses an internal early-stopping validation split after oversampling, so its early-stopping decision can see repeated rows. This does not put outer test rows into training, but is a limitation of the training procedure. Categories with very few examples have unstable scores, and the two people have different category and account coverage. The comparisons are exploratory; five correlated cross-validation folds and two people do not support a simple independent-fold t-test or a claim about all users.

**Near-duplicate descriptions can still cross folds.** Grouping only keeps *identical* descriptions together, so repeated merchants whose descriptions differ only by a store number, date, or reference code can appear in both training and test folds, which may make within-person scores somewhat optimistic.

Code: [ablation runner](../ml/feature_ablation.py) and [tuned comparison](../ml/model_comparison.py). Raw transactions and per-row predictions are not included in these result files.

To rerun with private labeled CSVs, run `python -m ml.feature_ablation --output results/tables/new_ablation_scores.csv` and `python -m ml.model_comparison --output results/new_tuned_comparison`. The ablation runner saves aggregate results as it progresses; `--resume` should only be used with the same data and settings. Regenerate the accuracy, macro F1, and amount-effect figures from the saved tables with `python -m ml.make_figures` ([figure code](../ml/make_figures.py)).

## Summary of answers

1. **Within each person:** Adding amount improved macro F1 in 7 of 8 person/model comparisons. Its effect varies by model; see the [within-person feature comparison](#1-do-extra-features-help-within-each-person).
2. **Across people:** Adding amount improved macro F1 in 5 of 8 comparisons: all four when training on Savanna, one of four when training on Jillian. See the [cross-person comparison](#2-do-extra-features-transfer-across-people).
3. **Best model:** In the tuned within-person comparison, neural network had the highest accuracy and macro F1 for Savanna and TF-IDF KNN did for Jillian. See the [model comparison](#3-which-models-performed-best).

## Answers to the original project questions

**Which machine learning techniques most accurately categorize transactions into main categories and subcategories?** For main categories, the best tuned model differs by person: neural network for Savanna (accuracy 0.657, macro F1 0.542) and TF-IDF KNN for Jillian (0.745, 0.623), against most-common baselines of 0.348 and 0.534 accuracy. The same model led on both metrics for each person. These experiments did not evaluate subcategories or train a model on a combined dataset. [See the tuned scores](tables/tuned_comparison_nov19_scores.csv).

**Can a model trained on one person's transactions categorize another person's transactions?** Not reliably. Training on Savanna and testing on Jillian, no tuned model beat the most-common baseline's accuracy (best 0.528 vs. 0.534), although macro F1 was higher (best 0.286 vs. 0.070), so the models recover some less common categories while missing more of the dominant one. Training on Jillian and testing on Savanna, embedding KNN reached 0.450 / 0.179 versus the baseline's 0.348 / 0.040. Different category and account coverage, and data from only two people, limit generalization. [See the cross-person scores](tables/tuned_comparison_nov19_cross_person.csv).
