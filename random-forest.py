import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("nearest-earth-objects(1910-2024).csv")

print(df.head())

X = df[['absolute_magnitude', 'estimated_diameter_min', 'estimated_diameter_max', 'orbiting_body', 'relative_velocity', 'miss_distance']]
y = df['is_hazardous']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.25)

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
    ('forest', RandomForestClassifier(random_state=0))
])

param_grid = {
    'forest__n_estimators': [10, 100, 500, 1000, None],
    'forest__max_depth': [3, 5, 7, 10, None],
    'forest__min_samples_split': [2, 5, 10],
    'forest__min_samples_leaf': [1, 2, 4]
}

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

accuracy = pipeline.score(X_test, y_test)
print(f'Accuracy: {accuracy:.4f}')

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best accuracy:", grid_search.best_score_)

