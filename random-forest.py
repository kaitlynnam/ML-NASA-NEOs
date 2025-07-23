import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
#from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE

df = pd.read_csv("nearest-earth-objects(1910-2024).csv")

print(df.head())

X = df[['absolute_magnitude', 'estimated_diameter_min', 'estimated_diameter_max', 'orbiting_body', 'relative_velocity', 'miss_distance']]
y = df['is_hazardous']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)

smote = SMOTE(random_state=0)

'''X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)'''

numerical_cols = ['absolute_magnitude', 'estimated_diameter_min', 'estimated_diameter_max', 'relative_velocity', 'miss_distance']
categorial_cols = ['orbiting_body']

print(X.isnull().sum())

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='mean'))
    ]), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorial_cols)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE()),
    ('forest', RandomForestClassifier(random_state=0))
])

param_grid = {
    'forest__n_estimators': [10, 100, 500],
    'forest__max_depth': [3, 5, 7, 10],
    'forest__min_samples_split': [2, 5, 10],
    'forest__min_samples_leaf': [1, 2, 4]
}

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

accuracy = pipeline.score(X_test, y_test)
print(f'Accuracy: {accuracy:.4f}')

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
y_proba = pipeline.predict_proba(X_test)[:, 1]
print("ROC AUC Score:", roc_auc_score(y_test, y_proba))

# plot confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best accuracy:", grid_search.best_score_)

forest = pipeline.named_steps['forest'] # or pipeline[-1]

importances = forest.feature_importances_

feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()

feat_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(feat_importance_df['Feature'], feat_importance_df['Importance'])
plt.gca().invert_yaxis()
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# plot train test class distribution
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
sns.countplot(x=y_train, ax=ax[0])
ax[0].set_title("Class Distribution (Train)")
sns.countplot(x=y_test, ax=ax[1])
ax[1].set_title("Class Distribution (Test)")
plt.tight_layout()
plt.show()

