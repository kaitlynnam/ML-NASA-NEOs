# ML-NASA-NEOs
# Goal
Determine weather Near Earth Objects (NEOs) are potentially hazardous to the orbiting body using NASA's NEO dataset.

# Model Comparison

| Model             | ROC AUC | Accuracy | F1 (hazardous) | Recall (hazardous) | Notes                          |
|:------------------|:-------:|:--------:|:--------------:|:------------------:|-------------------------------|
| Logistic Regression (no grid search) | 0.947   | 91.42%      | 0.67            | 0.67               | Simple, interpretable baseline |
| Random Forest      | 0.8354848084004779 | 73.65%     | 0.88           | 0.46               | Handles nonlinearity better    |
| XGBoost (Calibrated)| 0.9288560826406859|   82.67%   | 0.60       | 0.68          | Tuned scale_pos_weight + calibration + threshold tuning |


# Metrics Explained
 * ROC AUC -
 * Accuracy -
 * F1 (hazardous) -
 * Recall (hazardous) -
  

# Final Takeaway
