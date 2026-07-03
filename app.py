import os
import streamlit as st
import requests

from predict import predict as local_predict


API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Loan Approval Predictor", layout="centered")

st.title("Loan Approval Prediction")
st.write("Enter applicant details to predict loan approval probability.")

with st.form("applicant_form"):
    gender = st.selectbox("Gender", ["Male", "Female"], index=0)
    married = st.selectbox("Married", ["Yes", "No"], index=0)
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"], index=0)
    education = st.selectbox("Education", ["Graduate", "Not Graduate"], index=0)
    self_emp = st.selectbox("Self Employed", ["No", "Yes"], index=0)
    applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, value=100)
    loan_term = st.number_input("Loan Amount Term (days)", min_value=0, value=360)
    credit_history = st.selectbox("Credit History", ["1.0", "0.0"], index=0)
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"], index=0)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_dict = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents if dependents != "3+" else "3",
        "Education": education,
        "Self_Employed": self_emp,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area,
    }

    # Try backend first
    try:
        resp = requests.post(f"{API_URL}/predict", json=input_dict, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            label = data.get("label")
            proba = data.get("probability")
            st.success(f"Prediction: {label} — Probability of approval: {proba:.2%}")
        else:
            st.warning("Backend returned an error; falling back to local prediction.")
            pred, proba = local_predict(input_dict)
            label = "Approved" if pred == 1 else "Rejected"
            st.success(f"Prediction (local): {label} — Probability of approval: {proba:.2%}")
    except Exception as e:
        st.warning(f"Backend unreachable ({e}); using local model.")
        try:
            pred, proba = local_predict(input_dict)
            label = "Approved" if pred == 1 else "Rejected"
            st.success(f"Prediction (local): {label} — Probability of approval: {proba:.2%}")
        except Exception as ex:
            st.error(f"Local prediction failed: {ex}")
