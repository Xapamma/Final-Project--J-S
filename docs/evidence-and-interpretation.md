# Evidence and Interpretation

**Checkpoint:** November 19, 2026, by 11:59 PM Mountain Time  
**Team:** Savanna McLelland and Jillian Baker 
**Route:** C  
**Date:** November 17th, 2026

Add a new, dated entry to the root `README.md` of your team's GitHub repository. Link the code, results, and figures or tables that support your summary. Keep earlier entries, commit the update, and submit the repository URL with that commit in Canvas. This file is optional supporting material; preliminary or negative results are welcome.

## Evidence so far

- **Route C:** Show what the quality checks say about the dataset and what an ML result says about its usefulness for your question. Link both kinds of evidence.

The existing [Jillian quality report](../results/data_quality_jillian_nov12.md) examines 580 transactions from the collected data after 88 internal-transfer rows were removed. It reports no missing values, one possible duplicate, and no empty months for those 580 rows. For the model analysis, we use all 668 labeled transactions from the same collected data: 393 credit-card and 275 checking rows. This includes the 88 internal-transfer rows, labeled as **Transfer**. Ten entirely empty spreadsheet rows were excluded from the analysis. Food & Dining represents 46.4% of the 668 transactions and Transfer 17.2%. The quality report's checks apply to the 580 rows it examined, rather than all 668 model-input rows.

The [Savanna quality report](../results/data_quality_savanna.md) covers 998 labeled transactions and reports no missing values, 77 possible duplicates, and no empty months. Food & Dining accounts for 34.8% of rows. The duplicate flags may be real repeat purchases. Savanna's report could not reconstruct cleaning-step counts because it used a Streamlit export without the processor's `raw_description` column and the raw files were unavailable.

The tuned [within-person results](../results/tables/tuned_comparison_nov19_scores.csv) suggest the data is useful for predicting main categories. For Savanna, neural network achieved accuracy of 0.657 and macro F1 of 0.542, compared with 0.348 and 0.040 for the most-common baseline. For Jillian, embedding KNN achieved accuracy of 0.762 and macro F1 of 0.612, compared with 0.464 and 0.063 for the baseline. Macro F1 weights categories equally; scores are pooled over five description-grouped outer folds. See the [model code](../ml/model_comparison.py) and [final results](../results/final_results_nov19.md#3-which-models-performed-best).

In the tuned [cross-person comparison](../results/tables/tuned_comparison_nov19_cross_person.csv), TF-IDF KNN achieved accuracy / macro F1 of 0.422 / 0.229 when trained on Savanna and tested on Jillian, and 0.383 / 0.160 in the reverse direction. We highlight TF-IDF KNN because it had the highest observed macro F1 of the tuned model types in both directions; we identified it after evaluating the cross-person results. The baseline accuracies were 0.464 and 0.348, respectively. These results suggest partial transfer, with performance varying by direction; they do not evaluate subcategories or a model trained on a combined dataset.

## Interpretation and next step

- Describe one informative error, failure, tradeoff, or limitation. How does it affect your conclusion?

We did not have enough time to evaluate how well the models predict **subcategories**, even though they were part of our original project question. We focused on main categories to compare features, model types, and cross-person performance. The results cannot tell us which model works best for subcategories or whether it would reliably assign those detailed labels. See the [answers to the original questions](../results/final_results_nov19.md#answers-to-the-original-project-questions).

- What did the evidence change in your thinking, and what will you test or revise next?

The fixed-setting [feature comparison](../results/final_results_nov19.md#1-do-extra-features-help-within-each-person) suggests no single feature set works best for every model and person. Adding amount improved within-person macro F1 in 6 of 8 person/model comparisons, but the size and direction of the change varied. Cross-person effects also differed by model and training direction. Amount can help, but these overall scores do not establish that it resolves particular same-merchant errors or transfers consistently to others.

Next, we would evaluate subcategory predictions with the same held-out approach and report category-level performance, especially for rare categories. We would compare training with and without rare-category oversampling while retaining those categories, and inspect same-merchant errors to see whether amount helps distinguish them. We would also examine how including the 88 internal-transfer rows affects the results.
