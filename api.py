from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from predict import load_artifacts, preprocess_input
import traceback

app = FastAPI(title="Loan Approval Prediction API")


class Applicant(BaseModel):
    Gender: str
    Married: str
    Dependents: str
    Education: str
    Self_Employed: str
    ApplicantIncome: float
    CoapplicantIncome: float
    LoanAmount: float
    Loan_Amount_Term: float
    Credit_History: str
    Property_Area: str


# Load artifacts once at startup
try:
    MODEL, SCALER, ENCODERS = load_artifacts()
except Exception:
    MODEL = SCALER = ENCODERS = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_endpoint(applicant: Applicant):
    if MODEL is None or SCALER is None or ENCODERS is None:
        raise HTTPException(status_code=500, detail="Model artifacts not loaded")

    input_dict = applicant.dict()
    try:
        X = preprocess_input(input_dict, ENCODERS, SCALER)
        proba = float(MODEL.predict_proba(X)[0][1])
        pred = int(MODEL.predict(X)[0])
        label = "Approved" if pred == 1 else "Rejected"
        return {"prediction": pred, "label": label, "probability": proba}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
