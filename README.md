# Customer Churn Prediction

## Objective
Predict whether a customer will churn.

## Dataset
7,043 customers, 33 original features.

## Target
Churn Value

## Important preprocessing decisions
- Removed identifiers
- Removed constant columns
- Removed target-leaking columns
- Converted Total Charges to numeric
- Handled zero-tenure customers
- Stratified train/test split

## Models
- Using Logistic Regression first as a baseline
- Then Random Forest, tuned with RandomizedSearchCV
- Then XGBoost, also tuned with RandomizedSearchCV

## Results
- Logistic Regression Results: (Accuracy = 80.27%), (Precision = 64.29%), (Recall = 57.75%), (F1 Score = 60.85%)
- Tuned Random Forest (final, on untouched test set): (Accuracy = 77.00%), (Precision = 54.72%), (Recall = 77.54%), (F1 Score = 64.16%), Confusion matrix: [[795 240][84 290]]
- Tuned XGBoost (final, on untouched test set): F1 Score = 63.92%

## What I learned
- Use fit_transform for X_train but for X_test use transform because we do not want the encoder to learn a different category structure from the test set
- Logistic Regression shows 80% accuracy which looks good on paper but the results also show a 58% recall and 64% precision which is actually more important

- Threshold tuning should happen on out-of-fold predictions (cross_val_predict), not on the test set, otherwise you're leaking test info into model selection
Final untouched-test F1: 0.63925
So our XGBoost OOF estimate of 0.6564 did not fully carry over to unseen data. That's normal and exactly why we kept the test set untouched.
What we've learned
- Logistic: ~`0.651`
- Tuned RF: ~`0.652` OOF
- Tuned XGBoost: 0.639 on test
- XGBoost is not our final model.
- RF is currently our strongest candidate.
We're done with the model-family comparison. No more squeezing XGBoost.

- Final model: Random Forest. 77.5% recall means it catches most actual churners, at the cost of a decent number of false positives (240) - worth it if missing a churner costs more than a wasted retention offer.

## Feature importance (Random Forest)
- Contract - Month-to-month: 11.72%
- Tenure Months: 9.82%
- Total Charges: 7.98%
- Monthly Charges: 5.77%
- Online Security - No: 5.63%
- Tech Support - No: 5.33%

## What's left
- Maybe a tiny demo (Streamlit or just a CLI) if ever need to make this clickable instead of just scripts

## Running the final model
- 'src/s5_train_final_model.py' retrains the tuned RF on the training set and saves it to 'models/rf_pipeline.joblib'
- 'src/predict.py' loads that pipeline and prints a churn probability for an example customer (threshold 0.50, matching the tuned RF's best threshold from the OOF search)