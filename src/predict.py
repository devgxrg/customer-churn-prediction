import joblib
import pandas as pd

pipeline = joblib.load("../models/rf_pipeline.joblib")

THRESHOLD = 0.50


def predict_churn(customer: dict):
    row = pd.DataFrame([customer])
    prob = pipeline.predict_proba(row)[:, 1][0]
    return prob, int(prob >= THRESHOLD)


if __name__ == "__main__":
    example_customer = {
        "Gender": "Female",
        "Senior Citizen": "No",
        "Partner": "Yes",
        "Dependents": "No",
        "Phone Service": "Yes",
        "Multiple Lines": "No",
        "Internet Service": "Fiber optic",
        "Online Security": "No",
        "Online Backup": "Yes",
        "Device Protection": "No",
        "Tech Support": "No",
        "Streaming TV": "Yes",
        "Streaming Movies": "No",
        "Contract": "Month-to-month",
        "Paperless Billing": "Yes",
        "Payment Method": "Electronic check",
        "Latitude": 34.02,
        "Longitude": -118.29,
        "Tenure Months": 5,
        "Monthly Charges": 70.35,
        "Total Charges": 350.0,
        "CLTV": 3500,
    }

    prob, pred = predict_churn(example_customer)
    print(f"Churn probability: {prob:.3f}")
    print("Prediction:", "Churn" if pred else "No churn")
