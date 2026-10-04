import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from s1_load_clean_split import categorical_columns, numerical_columns, X_train, y_train

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
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

pipeline.fit(X_train, y_train)

joblib.dump(pipeline, "../models/rf_pipeline.joblib")
print("Saved pipeline to ../models/rf_pipeline.joblib")
