from sklearn.model_selection import train_test_split
import pandas as pd
data = pd.read_excel("../data/raw/Telco_customer_churn.xlsx")

df = pd.DataFrame(data)

df["Total Charges"] = pd.to_numeric(
    df["Total Charges"],
    errors="coerce"
)

target = "Churn Value"

drop_columns = ["CustomerID", "Count", "Country", "State", "City", "Zip Code", "Lat Long", "Churn Label", "Churn Score", "Churn Reason"]

y = df[target]
X = df.drop(columns = drop_columns + [target])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train.loc[X_train["Total Charges"].isna(), "Total Charges"] = 0
X_test.loc[X_test["Total Charges"].isna(), "Total Charges"] = 0

numerical_columns = ["Latitude", "Longitude", "Tenure Months", "Monthly Charges", "Total Charges", "CLTV"]
categorical_columns = ["Gender", "Senior Citizen", "Partner", "Dependents", "Phone Service", "Multiple Lines", "Internet Service", "Online Security",	"Online Backup", "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies", "Contract", "Paperless Billing", "Payment Method"]