import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv("nearest-earth-objects(1910-2024).csv")

print(df.head())

X = df[['absolute_magnitude', 'estimated_diameter_min', 'estimated_diameter_max', 'relative_velocity', 'miss_distance']]
y = df['is_hazardous']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)

print(X.isnull().sum())


pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=0)),
    ('logreg', LogisticRegression(random_state=0))
])

pipeline.fit(X_train, y_train)


accuracy = pipeline.score(X_test, y_test)

print(f"Accuracy: {accuracy * 100:.2f}%")

y_pred = pipeline.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


y_prob = pipeline.predict_proba(X_test)[:, 1] # gets predicted probability that the NEO is hazardous, [:,1] means it just grabs the probability it is hazardous
roc_auc = roc_auc_score(y_test, y_prob) # Measures how well the probabilities rank true positives over ture negitives

print(f"ROC AUC Score: {roc_auc:.4f}")

param_grid = {
    'logreg__penalty': ['l2'],
    'logreg__C': [0.01, 0.1, 1, 10, 100],
    'logreg__solver': ['lbfgs'],
    'logreg__max_iter': [100, 200, 500]
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

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(8,6))
plt.plot(recall, precision, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=grid_search.best_estimator_.named_steps['logreg'].classes_
)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()