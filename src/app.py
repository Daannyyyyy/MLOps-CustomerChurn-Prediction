from fastapi import FastAPI
import pandas as pd
import joblib
import os
from pydantic import BaseModel

from src.utils import preprocess_data

class ChurnInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

app = FastAPI()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "models", "churn_model.pkl"))
feature_names = joblib.load(os.path.join(BASE_DIR, "models", "feature_names.pkl"))


@app.get("/")
def home():
    return{"Message": "Churn Prediction app is running"}
@app.post("/predict")
def predict(data: ChurnInput):
    df = pd.DataFrame([data])
    df = preprocess_data(df)
    df = df.reindex(columns=feature_names, fill_value=0)

    prediction = model.predict(df)

    return {"churn Prediction" : prediction}

