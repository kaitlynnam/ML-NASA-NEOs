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

 * ROC AUC – Measures how well the model separates hazardous from non-hazardous asteroids across all thresholds.
Closer to 1 is better; 0.5 = random guessing. Above 0.9 = excellent.

 * Accuracy – Overall correctness of predictions.
Misleading with imbalanced data—can look high even if hazardous cases are missed.

 * F1 (Hazardous) – Balances precision and recall for hazardous class.
Useful when both false positives and false negatives matter.

 * Recall (Hazardous) – How many actual hazardous asteroids were correctly identified.
Crucial for safety-critical applications—higher recall = fewer missed threats.
  

# Final Takeaway

 * Logistic Regression delivers the highest ROC AUC and accuracy, making it a strong, simple baseline. But its recall and F1 for hazardous asteroids are moderate, meaning it misses some hazardous cases and isn’t great at balancing false positives and negatives.
   
 * Random Forest has a noticeably lower ROC AUC and accuracy but shines with the highest F1 score on hazardous asteroids. This means it’s better at balancing precision and recall on the critical class but misses more overall positives (lower recall).
   
 * XGBoost (Calibrated) hits a sweet spot—strong ROC AUC close to logistic regression, better recall than both models, and improved calibration + threshold tuning that helps catch more hazardous asteroids while controlling false alarms. However, its F1 is slightly lower than Random Forest, suggesting a trade-off in precision.

# WINNER: XGBoost!!!
If you’re trying to detect as many hazardous NEOs as possible while still maintaining strong overall performance, XGBoost wins. It’s the best choice when false negatives are unacceptable.
