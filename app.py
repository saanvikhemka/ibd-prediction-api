from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# Load trained model
model = joblib.load("ibd_conversion_model.pkl")

# Define input schema
class PredictionInput(BaseModel):
    sector: str
    revenue_band: str
    employee_band: str
    years_in_business: str
    new_to_internationalisation: str
    new_to_market: str
    is_sme: str
    enquiry_type: str

@app.get("/")
def root():
    return {"message": "IBD Conversion Prediction API is running"}

@app.post("/predict")
def predict(data: PredictionInput):
    input_df = pd.DataFrame([{
        "CRM_Organisation Sector": data.sector,
        "CRM_Organisation Revenue Band": data.revenue_band,
        "CRM_Organisation Employee Band": data.employee_band,
        "CRM_Organisation Years in Business": data.years_in_business,
        "CRM_IBD_New to Internationalisation?": data.new_to_internationalisation,
        "CRM_IBD_L2_New to Market?": data.new_to_market,
        "CRM_IBD_L2_Is company a SME?": data.is_sme,
        "CRM_IBD Enquiry Type": data.enquiry_type
    }])

    prediction = model.predict(input_df)[0]
    prediction = max(0.0, min(1.0, float(prediction)))

    # Convert to category
    if prediction >= 0.90:
        category = "Successful - 100%"
    elif prediction >= 0.65:
        category = "High - 75%"
    elif prediction >= 0.40:
        category = "Medium - 50%"
    elif prediction >= 0.15:
        category = "Low - 25%"
    else:
        category = "Unsuccessful - 0%"

    return {
        "prediction_score": round(prediction, 4),
        "category": category
    }