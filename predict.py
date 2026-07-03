import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder


CAT_COLUMNS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]

NUM_COLUMNS = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
]


def load_artifacts(model_path="loan_approval_model.pkl", scaler_path="scaler.pkl", data_path="data/trainDataset.csv"):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Load dataset to rebuild label encoders the same way as in training
    df = pd.read_csv(data_path)
    df.drop("Loan_ID", axis=1, inplace=True)

    # Fill missing values same as notebook
    categorical_columns = ["Gender", "Married", "Dependents", "Self_Employed", "Credit_History"]
    for column in categorical_columns:
        df[column].fillna(df[column].mode()[0], inplace=True)

    numerical_columns = ["LoanAmount", "Loan_Amount_Term"]
    for column in numerical_columns:
        df[column].fillna(df[column].median(), inplace=True)

    # Fit label encoders per categorical column so mappings match training
    encoders = {}
    for col in CAT_COLUMNS + ["Loan_Status"]:
        le = LabelEncoder()
        df[col] = df[col].astype(str)
        le.fit(df[col])
        encoders[col] = le

    return model, scaler, encoders


def preprocess_input(input_dict, encoders, scaler):
    # Build DataFrame with one row
    df = pd.DataFrame([input_dict])

    # Ensure columns exist and types
    for col in NUM_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill na for categorical fields by using mode from encoder classes if needed
    for col in CAT_COLUMNS:
        df[col] = df[col].astype(str)
        le = encoders.get(col)
        if le is not None:
            # If unseen label, attempt to map using le.transform will fail; map to most frequent
            if df.at[0, col] not in list(le.classes_):
                df.at[0, col] = le.classes_[0]
            df[col] = le.transform(df[col])

    # Scale numerical columns
    df_scaled = df.copy()
    df_scaled[NUM_COLUMNS] = scaler.transform(df[NUM_COLUMNS])

    return df_scaled


def predict(input_dict):
    model, scaler, encoders = load_artifacts()
    X = preprocess_input(input_dict, encoders, scaler)
    proba = model.predict_proba(X)[0][1]
    pred = int(model.predict(X)[0])
    return pred, proba


if __name__ == "__main__":
    # quick manual test
    sample = {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "0",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 2000,
        "LoanAmount": 150,
        "Loan_Amount_Term": 360,
        "Credit_History": "1.0",
        "Property_Area": "Urban",
    }
    p, pr = predict(sample)
    print("Prediction:", p, "Prob:", pr)
