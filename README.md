# ML-NASA-NEOs
# Goal
Determine weather Near Earth Objects (NEOs) are potentially hazardous to the orbiting body using NASA's NEO dataset.

# Model Comparison

| Model             | ROC AUC | Accuracy | F1 (hazardous) | Recall (hazardous) | Notes                          |
|:------------------|:-------:|:--------:|:--------------:|:------------------:|-------------------------------|
| Logistic Regression (no grid search) | 0.947   | 91.42%      | 0.67            | 0.67               | Simple, interpretable baseline |
| Random Forest      | 0.835 | 73.65%     | 0.88           | 0.46               | Handles nonlinearity better    |
| XGBoost (Calibrated)| 0.928|   82.67%   | 0.60       | 0.68          | Tuned scale_pos_weight + calibration + threshold tuning |


# Metrics Explained

 * ROC AUC - Measures how well your model separates the two classes (hazardous vs non-hazardous) across all classification thresholds. The closer to 1, the better your model distinguishes hazardous asteroids from safe ones regardless of the decision cutoff. 0.5 means random guessing; above 0.9 is excellent.
   
 * Accuracy - The proportion of total predictions your model got right (both hazardous and non-hazardous). It’s easy to understand but can be misleading if your classes are imbalanced (which they are). High accuracy sounds good, but if hazardous asteroids are rare, your model might just be guessing “non-hazardous” a lot and still get high accuracy.
   
 * F1 (hazardous) - The harmonic mean of precision and recall specifically for the hazardous class. Balances false positives and false negatives — crucial when missing a hazardous asteroid (false negative) or over-warning (false positive) both have costs. Higher F1 means better overall detection of hazardous asteroids, balancing precision and recall.

 * Recall (hazardous) - The percentage of actual hazardous asteroids your model correctly identifies. Critical for safety — you want to catch as many hazardous asteroids as possible (minimize false negatives). High recall means fewer missed hazardous asteroids, even if you sometimes raise false alarms.
  

# Final Takeaway

 * Logistic Regression delivers the highest ROC AUC and accuracy, making it a strong, simple baseline. But its recall and F1 for hazardous asteroids are moderate, meaning it misses some hazardous cases and isn’t great at balancing false positives and negatives.
   
 * Random Forest has a noticeably lower ROC AUC and accuracy but shines with the highest F1 score on hazardous asteroids. This means it’s better at balancing precision and recall on the critical class but misses more overall positives (lower recall).
   
 * XGBoost (Calibrated) hits a sweet spot—strong ROC AUC close to logistic regression, better recall than both models, and improved calibration + threshold tuning that helps catch more hazardous asteroids while controlling false alarms. However, its F1 is slightly lower than Random Forest, suggesting a trade-off in precision.

