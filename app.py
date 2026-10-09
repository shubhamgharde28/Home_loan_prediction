from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path(__file__).resolve().parent / "train.csv"

CATEGORICAL_FEATURES = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]
NUMERIC_FEATURES = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
REQUIRED_COLUMNS = FEATURES + ["Loan_Status"]


def train_model(data_path: Path) -> Pipeline:
    """Train the notebook's tuned random-forest model on the supplied CSV."""
    data = pd.read_csv(data_path)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(data.columns))
    if missing_columns:
        raise ValueError(
            "The training CSV is missing required columns: "
            + ", ".join(missing_columns)
        )

    features = data[FEATURES]
    target = data["Loan_Status"].astype(str).str.strip().str.upper()
    unexpected_statuses = sorted(set(target.dropna()) - {"Y", "N"})
    if unexpected_statuses or target.isna().any():
        raise ValueError("Loan_Status must contain only non-empty Y or N values.")

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=101,
                    max_depth=3,
                    random_state=1,
                ),
            ),
        ]
    )
    model.fit(features, target)
    return model


@st.cache_resource(show_spinner="Training the loan prediction model...")
def load_model() -> Pipeline:
    return train_model(DATA_PATH)


def main() -> None:
    st.set_page_config(page_title="Home Loan Prediction", page_icon="🏠")
    st.title("Home Loan Eligibility Prediction")
    st.write(
        "Enter applicant details to see the prediction from a random-forest "
        "model trained on the project's sample data."
    )

    try:
        model = load_model()
    except (FileNotFoundError, ValueError) as error:
        st.error(f"Could not train the model: {error}")
        st.stop()

    with st.form("loan_application"):
        left, right = st.columns(2)

        with left:
            gender = st.selectbox("Gender", ["Male", "Female"])
            married = st.selectbox("Married", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
            education = st.selectbox("Education", ["Graduate", "Not Graduate"])
            self_employed = st.selectbox("Self-employed", ["No", "Yes"])
            property_area = st.selectbox(
                "Property area", ["Urban", "Semiurban", "Rural"]
            )

        with right:
            applicant_income = st.number_input(
                "Applicant monthly income",
                min_value=0,
                value=5000,
                step=500,
            )
            coapplicant_income = st.number_input(
                "Co-applicant monthly income",
                min_value=0,
                value=0,
                step=500,
            )
            loan_amount = st.number_input(
                "Loan amount (in thousands)",
                min_value=0,
                value=120,
                step=10,
            )
            loan_term = st.selectbox(
                "Loan term (months)",
                [12, 36, 60, 84, 120, 180, 240, 300, 360, 480],
                index=8,
            )
            credit_history = st.selectbox(
                "Credit history",
                options=[1, 0],
                format_func=lambda value: "Good" if value == 1 else "Poor",
            )

        submitted = st.form_submit_button("Predict eligibility")

    if submitted:
        application = pd.DataFrame(
            [
                {
                    "Gender": gender,
                    "Married": married,
                    "Dependents": dependents,
                    "Education": education,
                    "Self_Employed": self_employed,
                    "Property_Area": property_area,
                    "ApplicantIncome": applicant_income,
                    "CoapplicantIncome": coapplicant_income,
                    "LoanAmount": loan_amount,
                    "Loan_Amount_Term": loan_term,
                    "Credit_History": credit_history,
                }
            ],
            columns=FEATURES,
        )
        prediction = model.predict(application)[0]
        prediction_probabilities = model.predict_proba(application)[0]
        predicted_index = list(model.classes_).index(prediction)
        probability = prediction_probabilities[predicted_index]

        if prediction == "Y":
            st.success("Prediction: likely eligible")
        else:
            st.warning("Prediction: unlikely eligible")
        st.metric("Predicted class probability", f"{probability:.1%}")

    st.caption(
        "For demonstration only. This prediction is not a lending decision, "
        "financial advice, or a guarantee of loan approval."
    )


if __name__ == "__main__":
    main()
