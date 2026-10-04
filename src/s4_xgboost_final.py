import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, RandomizedSearchCV
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from s1_load_clean_split import categorical_columns, numerical_columns, X_train, y_train, X_test, y_test


preprocessor = ColumnTransformer([
    ("categorical",
     OneHotEncoder(),
     categorical_columns),
    ("numerical",
     StandardScaler(),
     numerical_columns)
])


model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
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

param_grid = {
    "model__n_estimators": [100, 200, 300],
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__max_depth": [3, 5, 7],
    "model__subsample": [0.8, 1.0],
    "model__colsample_bytree": [0.8, 1.0]
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

    f1 = f1_score(y_train, y_pred)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

best_model.fit(X_train, y_train)

y_prob_test = best_model.predict_proba(X_test)[:, 1]

y_pred_test = (y_prob_test >= best_threshold).astype(int)

print("Test F1:", f1_score(y_test, y_pred_test))
