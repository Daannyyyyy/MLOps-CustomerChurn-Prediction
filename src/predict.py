import joblib
import pandas as pd
from utils import preprocess_data


model = joblib.load("models/churn_model.pkl")
feature_names = joblib.load("models/feature_names.pkl")

customer = {
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "Yes",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.35,
  "TotalCharges": 845.5
}


df = pd.DataFrame(customer)
X = preprocess_data(df)
X = X.reindex(columns=feature_names, fill_value=0)

prediction = model.predict(X)
print("Prediction:", prediction)