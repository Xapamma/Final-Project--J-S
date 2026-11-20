# Evidence and Interpretation

**Checkpoint:** November 19, 2026, by 11:59 PM Mountain Time  
**Team:** Savanna McLelland and Jillian Baker  
**Route:** C  
**Date:** November 19, 2026

Add a new, dated entry to the root `README.md` of your team's GitHub repository. Link the code, results, and figures or tables that support your summary. Keep earlier entries, commit the update, and submit the repository URL with that commit in Canvas. This file is optional supporting material; preliminary or negative results are welcome.

## Evidence so far

- **Route C:** Show what the quality checks say about the dataset and what an ML result says about its usefulness for your question. Link both kinds of evidence.

The [Jillian quality report](../results/data_quality_jillian_nov12.md) checks the exact 580 transactions used by the models: 393 credit-card and 187 checking rows, after 88 internal transfers between Jillian's own accounts were removed. It reports no missing values, one possible duplicate (kept and flagged), and no empty months. Food & Dining is 53.4% of rows and the remaining credit-card-payment `Transfer` rows are 4.7%.

The [Savanna quality report](../results/data_quality_savanna.md) covers 998 labeled transactions and reports no missing values, 77 possible duplicates, and no empty months. Food & Dining accounts for 34.8% of rows. The duplicate flags may be real repeat purchases. Savanna's data contains no internal transfers between her own accounts; 18 credit-card payments remain labeled `Transfer`. Her report could not reconstruct cleaning-step counts because it used a Streamlit export without the processor's `raw_description` column and the raw files were unavailable, so how those transfers were removed is not recorded.

The tuned [within-person results](../results/tables/tuned_comparison_nov19_scores.csv) suggest the data is useful for predicting main categories. For Savanna, the neural network had the highest accuracy and macro F1 (0.657 and 0.542), compared with 0.348 and 0.040 for the most-common baseline. For Jillian, TF-IDF KNN had the highest accuracy and macro F1 (0.745 and 0.623), compared with 0.534 and 0.070 for the baseline. Macro F1 weights categories equally; scores are pooled over five description-grouped outer folds. See the [model code](../ml/model_comparison.py), [figures](../results/media/macro_f1_comparison.svg), and [final results](../results/final_results_nov19.md#3-which-models-performed-best).

The tuned [cross-person comparison](../results/tables/tuned_comparison_nov19_cross_person.csv) shows that a model trained on one person does not reliably categorize the other's transactions. Training on Savanna and testing on Jillian, no model beat the most-common baseline's accuracy (best 0.528 vs. 0.534); TF-IDF KNN had the highest macro F1 (0.286 vs. 0.070). Training on Jillian and testing on Savanna, embedding KNN had the highest accuracy and macro F1 (0.450 / 0.179 vs. the baseline's 0.348 / 0.040). The best model differs by direction and was identified after seeing the results. These experiments do not evaluate subcategories or a model trained on a combined dataset.

## Interpretation and next step

- Describe one informative error, failure, tradeoff, or limitation. How does it affect your conclusion?

**Training on Savanna and testing on Jillian, no tuned model beat the most-common baseline's accuracy.** Always guessing Food & Dining is right for 53.4% of Jillian's transactions. The best tuned model, embedding KNN, reached 0.528 accuracy, and TF-IDF KNN reached 0.488. Their macro F1 was much higher than the baseline's (0.286 for TF-IDF KNN vs. 0.070), so the models do recognize some of Jillian's less common categories. But they miss enough of her dominant category that overall accuracy falls below the baseline. A plausible cause is that the two datasets differ: Food & Dining is 35% of Savanna's rows but 53% of Jillian's, and the two of us use different banks, accounts, and merchants. A model trained on Savanna's data has learned her mix of categories and her merchant descriptions, not Jillian's.

This changes our answer to the second project question. Judged by macro F1 alone, transfer between people looks partially successful. Judged by accuracy, a model trained on Savanna is no more useful for Jillian than a constant guess. In the other direction, training on Jillian, every model beat the baseline (best: embedding KNN, 0.450 / 0.179 vs. 0.348 / 0.040). So our conclusion is that a model trained on one person does not reliably categorize another person's transactions, and how well it does depends on whose data it was trained on. See the [cross-person table](../results/tables/tuned_comparison_nov19_cross_person.csv).

- What did the evidence change in your thinking, and what will you test or revise next?

We started out looking for one best model and expected the amount to help broadly. The evidence did not support either expectation. The best tuned model differs by person: a neural network for Savanna (0.657 / 0.542) and TF-IDF KNN for Jillian (0.745 / 0.623). Adding the amount improved within-person macro F1 in 7 of 8 comparisons, but across people it helped all four models when training on Savanna and only one of four when training on Jillian ([feature comparison](../results/final_results_nov19.md#1-do-extra-features-help-within-each-person)). Amount carries real signal within one person's spending patterns, but that signal does not reliably carry over to someone else's. Tuning and rare-category oversampling mattered more than we expected. On the same 580 Jillian transactions, our untuned Nov 12 embedding KNN scored 0.586 / 0.385.

Next, we will:

1. **Train on the combined dataset**, evaluated on each person's held-out folds, to see whether pooling closes the cross-person gap.
2. **Check per-category results for same-merchant cases**, such as Chevron and Maverik snacks vs. fuel, with and without amount, to test whether amount is what separates them.
3. **Evaluate subcategory predictions** with the same held-out approach, since we only tested main categories.
