import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from s1_load_clean_split import categorical_columns, numerical_columns, X_train, y_train, X_test, y_test

preprocessor = ColumnTransformer([
    ("categorical",
     OneHotEncoder(),
     categorical_columns),
    ("numerical",
     StandardScaler(),
     numerical_columns)
])

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

scores = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring="f1"
)

y_prob_oof = cross_val_predict(
    pipeline,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba"
)[:, 1]

param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__max_depth": [None, 5, 10, 20],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4]
}

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=10,
    cv=cv,
    scoring="f1",
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)
best_model = search.best_estimator_

rf_model = best_model.named_steps["model"]
preprocessor = best_model.named_steps["preprocessor"]

feature_names = preprocessor.get_feature_names_out()
importances = rf_model.feature_importances_

feature_importance = sorted(
    zip(feature_names, importances),
    key=lambda x: x[1],
    reverse=True
)

for feature, importance in feature_importance[:15]:
    print(feature, importance)

y_prob_oof_tuned = cross_val_predict(
    best_model,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba"
)[:, 1]

best_f1 = 0
best_threshold = 0

for threshold in np.arange(0.10, 0.91, 0.01):
    y_pred = (y_prob_oof_tuned >= threshold).astype(int)

    precision = precision_score(y_train, y_pred)
    recall = recall_score(y_train, y_pred)
    f1 = f1_score(y_train, y_pred)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

best_model.fit(X_train, y_train)

y_prob_test = best_model.predict_proba(X_test)[:, 1]

y_pred_test = (y_prob_test >= best_threshold).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred_test))
print("Precision:", precision_score(y_test, y_pred_test))
print("Recall:", recall_score(y_test, y_pred_test))
print("F1:", f1_score(y_test, y_pred_test))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_test))

# print("Best threshold:", best_threshold)
# print("Best F1:", best_f1)
