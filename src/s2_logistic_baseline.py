import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from s1_load_clean_split import categorical_columns, numerical_columns, X, X_train, X_test, y, y_train, y_test

preprocessor = ColumnTransformer([
    ("categorical",
     OneHotEncoder(),
     categorical_columns),
    ("numerical",
     StandardScaler(),
     numerical_columns)
])

thresholds = np.arange(0.20, 0.60, 0.01)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

model = LogisticRegression(max_iter=1000, class_weight="balanced")

model.fit(X_train_processed, y_train)
y_prob = model.predict_proba(X_test_processed)[:, 1]


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

print(scores)
print(scores.mean())

y_prob_oof = cross_val_predict(
    pipeline,
    X_train,
    y_train,
    cv=cv,
    method="predict_proba"
)[:, 1]

for threshold in thresholds:
    y_pred_threshold = (y_prob_oof >= threshold).astype(int)

    precision = precision_score(y_train, y_pred_threshold)
    recall = recall_score(y_train, y_pred_threshold)
    f1 = f1_score(y_train, y_pred_threshold)

    print(threshold, precision, recall, f1)
