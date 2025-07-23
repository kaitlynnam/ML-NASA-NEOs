import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import joblib

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay


df = pd.read_csv("nearest-earth-objects(1910-2024).csv")

print(df.head())

X = df[['absolute_magnitude', 'estimated_diameter_min', 'estimated_diameter_max', 'relative_velocity', 'miss_distance']]
y = df['is_hazardous']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)

print(X.isnull().sum())

spw = 73754 / 10796

pipeline = Pipeline([
    ('imputer', SimpleImputer()),
    #('smote', SMOTE(random_state=0)),
    ('xgb', XGBClassifier(scale_pos_weight=spw))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

accuracy = pipeline.score(X_test, y_test)
print(f'Accuracy: {accuracy:.4f}')

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
y_proba = pipeline.predict_proba(X_test)[:, 1]
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))



param_grid = {
    'xgb__max_depth': [3, 6, 9],
    'xgb__learning_rate': [0.01, 0.1, 0.2],
    'xgb__n_estimators': [100, 200, 300],
    'xgb__subsample': [0.7, 1],
    'xgb__colsample_bytree': [0.7, 1]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best ROC AUC:", grid_search.best_score_)

accuracy = grid_search.best_estimator_.score(X_test, y_test)

print(f"Accuracy: {accuracy * 100:.2f}%")

y_prob = grid_search.best_estimator_.predict_proba(X_test)[:, 1]
y_pred = grid_search.best_estimator_.predict(X_test)
roc_auc = roc_auc_score(y_test, y_prob)

best_xgb = grid_search.best_estimator_.named_steps['xgb']

imputer = grid_search.best_estimator_.named_steps['imputer']
X_train_imputed = imputer.transform(X_train)
X_test_imputed = imputer.transform(X_test)

calibrated_model = CalibratedClassifierCV(estimator=best_xgb, method='sigmoid', cv=5)
calibrated_model.fit(X_train_imputed, y_train)

# Predict calibrated probabilities
y_probs_calibrated = calibrated_model.predict_proba(X_test_imputed)[:, 1]
roc_auc_calibrated = roc_auc_score(y_test, y_probs_calibrated)
print("Calibrated ROC AUC:", roc_auc_calibrated)

precision, recall, thresholds = precision_recall_curve(y_test, y_probs_calibrated)
f1_scores = 2 * recall * precision / (recall + precision + 1e-8)
best_threshold = thresholds[np.argmax(f1_scores)]
print(f"Best threshold based on F1: {best_threshold:.3f}")

y_pred_best = (y_probs_calibrated >= best_threshold).astype(int)
print(classification_report(y_test, y_pred_best))

fpr, tpr, _ = roc_curve(y_test, y_prob)
fpr_cal, tpr_cal, _ = roc_curve(y_test, y_probs_calibrated)

plt.plot(fpr, tpr, label='Uncalibrated')
plt.plot(fpr_cal, tpr_cal, label='Calibrated')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


disp = PrecisionRecallDisplay.from_predictions(y_test, y_probs_calibrated)
disp.ax_.set_title("Precision-Recall Curve (Calibrated)")
plt.show()

importances = best_xgb.feature_importances_
features = X.columns

sns.barplot(x=importances, y=features)
plt.title("XGBoost Feature Importances")
plt.show()

cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Best Threshold)')
plt.show()

joblib.dump((calibrated_model, best_threshold), 'calibrated_xgb_model.pkl')