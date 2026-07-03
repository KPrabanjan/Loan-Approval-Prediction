### Loan Approval Prediction using Machine Learning

Project Overview

This project builds a supervised machine learning model to predict whether a loan application will be approved or rejected based on applicant information such as income, education, employment status, credit history, and property area.

The project demonstrates a complete end-to-end machine learning workflow, including data preprocessing, exploratory data analysis, handling class imbalance, model training, evaluation, business interpretation, and prediction on new applicants.

### Objectives

* Perform exploratory data analysis (EDA).

* Handle missing values and clean the dataset.

* Encode categorical variables.

* Scale numerical features.


### Run the Streamlit Frontend

1. Activate the project's virtual environment.

```powershell
cd "C:\Prab\vs code\LoanApprovalPrediction"
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The Streamlit app provides a simple form to enter applicant details and returns the predicted approval label and probability.

### Backend API (FastAPI)

Start the backend locally (after activating the venv):

```powershell
cd "C:\Prab\vs code\LoanApprovalPrediction"
.\venv\Scripts\Activate.ps1
uvicorn api:app --reload --port 8000
```

Endpoints:
- `GET /health` — returns service status
- `POST /predict` — JSON body with applicant fields, returns `prediction`, `label`, `probability`

Docker:

```powershell
docker build -t loan-approval-backend .
docker run -p 8000:8000 loan-approval-backend
```
* Handle class imbalance using SMOTE.

* Train and evaluate multiple classification models.

* Compare model performance using Accuracy, Precision, Recall, F1-score, and ROC-AUC.

* Provide business-oriented insights for loan approval decisions.

### Dataset

Source: Kaggle - Loan Prediction Dataset

Records: 614 loan applications

Target Variable: Loan_Status

Features Include:

* Gender

* Married

* Dependents

* Education

* Self_Employed

* ApplicantIncome

* CoapplicantIncome

* LoanAmount

* Loan_Amount_Term

* Credit_History

* Property_Area

### Technologies Used

| Category           | Tools                    |
| ------------------ | ------------------------ |
| Language           | Python                   |
| Data Processing    | Pandas, NumPy            |
| Visualization      | Matplotlib, Seaborn      |
| Machine Learning   | Scikit-learn             |
| Imbalance Handling | SMOTE (imbalanced-learn) |
| Model Persistence  | Joblib                   |

### Project Workflow

Data Loading

Exploratory Data Analysis (EDA)

Missing Value Handling

Feature Encoding

Feature Scaling

Train-Test Split

SMOTE Oversampling

Logistic Regression

Decision Tree

Random Forest

Model Comparison

Feature Importance Analysis

Business Interpretation

### Models Evaluated

| Model               | Purpose                     |
| ------------------- | --------------------------- |
| Logistic Regression | Baseline linear classifier  |
| Decision Tree       | Non-linear rule-based model |
| Random Forest       | Ensemble tree-based model   |

### Final Results

| Model               | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ------------------- | -------- | --------- | ------ | -------- | ------- |
| Logistic Regression | 0.8130   | 0.8444    | 0.8941 | 0.8686   | 0.7628  |
| Decision Tree       | 0.7642   | 0.8784    | 0.7647 | 0.8176   | 0.7639  |
| Random Forest       | 0.7967   | 0.8488    | 0.8588 | 0.8538   | 0.7584  |

### Best Model

### Logistic Regression

Selected

Selected based on highest Accuracy, Recall, and F1-score.

### Business Insights

* Credit History is the most influential factor in loan approval decisions.

* Applicant Income and Loan Amount also significantly impact approval probability.

* The model can assist banks in reducing manual screening effort.

* A deployment threshold of 0.50 provides a reasonable balance between precision and recall.

### Project Structure

LoanApprovalPrediction/

│

├── data/

│ └── train.csv

│

├── report/

│

├── LoanPrediction.ipynb

├── loan_approval_model.pkl

├── scaler.pkl

├── README.md

├── requirements.txt

└── .gitignore

### How to Run

1

Clone the repository

git clone <repository-url>

2

Install dependencies

pip install -r requirements.txt

3

Open the notebook

jupyter notebook LoanPrediction.ipynb

4

Run all cells

Execute the notebook from top to bottom.

### Future Improvements

* Hyperparameter tuning using GridSearchCV.

* Cross-validation for more robust evaluation.

* Deployment using Flask or Streamlit.

* Integration with a web-based loan application form.

* Use of larger real-world banking datasets.

### Author

PRABANJAN K
AI & Data Science Student
Chennai Institute of Technology
