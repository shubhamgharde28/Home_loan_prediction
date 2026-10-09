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


@st.cache_data
def load_training_data(data_path: Path) -> pd.DataFrame:
    """Load the sample data used to train the prediction model."""
    return pd.read_csv(data_path)


def render_overview(data: pd.DataFrame) -> None:
    """Show summary metrics and charts for the training data."""
    total_applications = len(data)
    approval_rate = data["Loan_Status"].astype(str).str.strip().str.upper().eq("Y").mean()
    median_loan = data["LoanAmount"].median()
    average_income = data["ApplicantIncome"].mean()

    metric_columns = st.columns(4)
    metric_columns[0].metric("Applications", f"{total_applications:,}")
    metric_columns[1].metric("Approval rate", f"{approval_rate:.1%}")
    metric_columns[2].metric(
        "Median loan amount", f"{median_loan:,.0f}" if pd.notna(median_loan) else "N/A"
    )
    metric_columns[3].metric(
        "Average applicant income",
        f"{average_income:,.0f}" if pd.notna(average_income) else "N/A",
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Loan application outcomes")
        status_counts = (
            data["Loan_Status"]
            .astype(str)
            .str.strip()
            .str.upper()
            .map({"Y": "Eligible", "N": "Not eligible"})
            .value_counts()
            .reindex(["Eligible", "Not eligible"], fill_value=0)
        )
        st.bar_chart(status_counts, y_label="Applications")

        st.subheader("Approval rate by property area")
        approval_by_area = (
            data.assign(
                Approved=data["Loan_Status"]
                .astype(str)
                .str.strip()
                .str.upper()
                .eq("Y")
            )
            .groupby("Property_Area", dropna=False)["Approved"]
            .mean()
            .sort_values(ascending=False)
        )
        st.bar_chart(approval_by_area, y_label="Approval rate")

    with right:
        st.subheader("Applicant income distribution")
        income = pd.to_numeric(data["ApplicantIncome"], errors="coerce").dropna()
        if income.empty:
            st.info("No applicant income values are available to chart.")
        else:
            income_bins = pd.cut(income, bins=10, include_lowest=True)
            income_counts = income_bins.value_counts().sort_index()
            income_counts.index = income_counts.index.astype(str)
            st.bar_chart(income_counts, y_label="Applications")

        st.subheader("Approval rate by credit history")
        approval_by_credit = (
            data.assign(
                Credit_History=data["Credit_History"].map(
                    {1: "Good credit history", 0: "Poor credit history"}
                ),
                Approved=data["Loan_Status"]
                .astype(str)
                .str.strip()
                .str.upper()
                .eq("Y"),
            )
            .dropna(subset=["Credit_History"])
            .groupby("Credit_History")["Approved"]
            .mean()
            .sort_values(ascending=False)
        )
        st.bar_chart(approval_by_credit, y_label="Approval rate")


def main() -> None:
    st.set_page_config(page_title="Home Loan Prediction", page_icon="🏠")
    st.title("Home Loan Eligibility Prediction")
    st.write(
        "Explore the project sample data, see its trends, and try the random-forest "
        "loan eligibility model."
    )

    try:
        model = load_model()
        data = load_training_data(DATA_PATH)
    except (FileNotFoundError, ValueError) as error:
        st.error(f"Could not load the loan prediction app: {error}")
        st.stop()

    overview_tab, prediction_tab, data_tab = st.tabs(
        ["Overview & graphs", "Predict eligibility", "Explore all data"]
    )

    with overview_tab:
        st.header("Loan application overview")
        st.caption("Charts and summary statistics are based on the training sample.")
        render_overview(data)

    with prediction_tab:
        st.header("Try a loan eligibility prediction")
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

    with data_tab:
        st.header("Complete training dataset")
        st.write(
            f"All {len(data):,} sample training records and their original columns."
        )
        st.download_button(
            "Download training data as CSV",
            data=data.to_csv(index=False).encode("utf-8"),
            file_name="home_loan_training_data.csv",
            mime="text/csv",
        )
        st.dataframe(data, hide_index=True)

    st.caption(
        "For demonstration only. This prediction is not a lending decision, "
        "financial advice, or a guarantee of loan approval."
    )


if __name__ == "__main__":
    main()
