# Evidence and Interpretation

**Checkpoint:** November 19, 2026, by 11:59 PM Mountain Time  
**Team:** Savanna McLelland and Jillian Baker 
**Route:** C  
**Date:** November 17th, 2026

Add a new, dated entry to the root `README.md` of your team's GitHub repository. Link the code, results, and figures or tables that support your summary. Keep earlier entries, commit the update, and submit the repository URL with that commit in Canvas. This file is optional supporting material; preliminary or negative results are welcome.

## Evidence so far

- **Route C:** Show what the quality checks say about the dataset and what an ML result says about its usefulness for your question. Link both kinds of evidence.

The quality checks show that both datasets have complete labels and no missing values in the reported fields, but their categories are unevenly represented. The [Jillian quality report](../results/data_quality_jillian.md) describes 393 labeled credit-card transactions, with Food & Dining making up 62% of them. The [Savanna quality report](../results/data_quality_savanna.md) describes 998 labeled transactions across seven accounts, with Food & Dining making up 35%. The reports flag one possible duplicate for Jillian and 77 for Savanna; these are possible repeat purchases and were not automatically removed. Neither report found a month without transactions. Savanna's dataset came from the Streamlit app's export rather than the processor output used for the cleaning-step checks. That export lacked the `raw_description` column expected by the quality-check script, and the raw files were unavailable when the report was generated, so the report could not reconstruct how many rows each cleaning step affected.

The [November 12 Jillian quality report](../results/data_quality_jillian_nov12.md) describes a later 580-row dataset that includes checking transactions. The [November 19 model results](../results/final_results_nov19.md) instead report 393 Jillian rows and 998 Savanna rows. These are different Jillian dataset versions, so the 580-row quality findings should not be treated as checks of the 393-row model input.

The model results suggest the labeled data contains useful information for predicting **main categories** within each person. In the tuned comparison, Savanna's neural network achieved accuracy of 0.657 and macro F1 of 0.542, compared with 0.348 and 0.040 for the most-common-category baseline. Jillian's embedding KNN achieved accuracy of 0.623 and macro F1 of 0.454, compared with 0.623 and 0.077 for the baseline. Macro F1 is useful here because it gives each category equal weight rather than letting the most common category dominate. These scores come from pooled predictions across five description-grouped outer folds. See the [model code](../ml/model_comparison.py), [score table](../results/tables/tuned_comparison_nov19_scores.csv), and [final results](../results/final_results_nov19.md#3-which-models-performed-best).

Cross-person results show more limited usefulness: tuned TF-IDF KNN achieved macro F1 of 0.290 when trained on Savanna and tested on Jillian, and 0.146 in the reverse direction; its accuracy was 0.562 and 0.359, respectively. The most-common baseline's accuracy was 0.623 and 0.348, respectively. The [cross-person table](../results/tables/tuned_comparison_nov19_cross_person.csv) and [final comparison](../results/final_results_nov19.md#2-do-extra-features-transfer-across-people) show that transfer depends on direction and is not consistently stronger than the baseline on accuracy. The experiments do not evaluate subcategories or a model trained on a combined dataset.

## Interpretation and next step

- Describe one informative error, failure, tradeoff, or limitation. How does it affect your conclusion?

We did not have enough time to evaluate how well the models predict **subcategories**, even though they were part of our original project question. We focused the experiments on main categories so we could compare features, model types, and cross-person performance. The results show that the dataset is useful for testing main-category predictions, but they cannot tell us which model works best for subcategories or whether a model would reliably assign those more detailed labels. See the [answers to the original questions](../results/final_results_nov19.md#answers-to-the-original-project-questions).

- What did the evidence change in your thinking, and what will you test or revise next?

The feature comparison made us less confident that one set of inputs or one model will work best for everyone. Adding amount improved within-person macro F1 in seven of eight person/model comparisons, but it reduced Savanna's random-forest score; type and account also had mixed effects. Cross-person scores were weaker and varied by training direction. This suggests that amount can be helpful, but the results do not yet show that it resolves specific same-merchant errors or transfers consistently between people. See the [feature comparisons](../results/final_results_nov19.md#1-do-extra-features-help-within-each-person) and [cross-person results](../results/final_results_nov19.md#2-do-extra-features-transfer-across-people).

Next, we would evaluate subcategory predictions with the same held-out approach and report category-level performance, especially where examples are rare. We would also compare training with and without rare-category oversampling, while keeping those categories, and inspect the same-merchant errors to see whether amount actually helps distinguish them. Before comparing dataset versions, we would reconcile Jillian's 393-row model input with the 580-row version described in the November 12 quality report.
