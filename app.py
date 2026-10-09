from pathlib import Path
import math

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path(__file__).resolve().parent / "train.csv"
TEST_DATA_PATH = Path(__file__).resolve().parent / "test.csv"

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


@st.cache_data
def load_test_data(data_path: Path) -> pd.DataFrame:
    """Load the unlabeled sample used for the notebook's test-data charts."""
    data = pd.read_csv(data_path)
    missing_columns = sorted(set(FEATURES) - set(data.columns))
    if missing_columns:
        raise ValueError(
            "The test CSV is missing required columns: "
            + ", ".join(missing_columns)
        )
    return data


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


def render_histogram(
    data: pd.DataFrame,
    column: str,
    title: str,
    key: str,
    *,
    density: bool = False,
    log_scale: bool = False,
) -> None:
    """Render a numeric distribution chart, skipping only empty series."""
    values = pd.to_numeric(data[column], errors="coerce").dropna()
    if log_scale:
        values = values[values > 0].map(math.log)
    if values.empty:
        st.info(f"No values are available for {title.lower()}.")
        return

    chart_data = pd.DataFrame({column: values})
    chart = px.histogram(
        chart_data,
        x=column,
        nbins=30,
        histnorm="probability density" if density else None,
        title=title,
        labels={column: "Natural log of value" if log_scale else column},
    )
    chart.update_layout(yaxis_title="Density" if density else "Applications")
    st.plotly_chart(chart, width="stretch", key=key)


def render_box_plot(
    data: pd.DataFrame,
    column: str,
    title: str,
    key: str,
    *,
    group_by: str | None = None,
) -> None:
    """Render a numeric box plot, optionally grouped by a category."""
    chart_data = data.copy()
    chart_data[column] = pd.to_numeric(chart_data[column], errors="coerce")
    chart_data = chart_data.dropna(subset=[column])
    if chart_data.empty:
        st.info(f"No values are available for {title.lower()}.")
        return

    chart = px.box(
        chart_data,
        x=group_by,
        y=column,
        title=title,
        points=False,
    )
    st.plotly_chart(chart, width="stretch", key=key)


def render_approval_chart(data: pd.DataFrame, column: str, key: str) -> None:
    """Show the eligible and not-eligible shares for each category."""
    chart_data = data[[column, "Loan_Status"]].copy()
    chart_data["Loan_Status"] = (
        chart_data["Loan_Status"].astype(str).str.strip().str.upper()
    )
    chart_data["Loan_Status"] = chart_data["Loan_Status"].map(
        {"Y": "Eligible", "N": "Not eligible"}
    )
    chart_data[column] = chart_data[column].fillna("Missing").astype(str)
    chart_data = chart_data.dropna(subset=["Loan_Status"])
    if chart_data.empty:
        st.info(f"No labeled applications are available for {column}.")
        return

    proportions = pd.crosstab(
        chart_data[column], chart_data["Loan_Status"], normalize="index"
    ).reindex(columns=["Eligible", "Not eligible"], fill_value=0)
    chart = px.bar(
        proportions,
        x=proportions.index,
        y=proportions.columns,
        barmode="stack",
        title=f"Loan outcome by {column.replace('_', ' ')}",
        labels={"x": column.replace("_", " "), "y": "Share of applications"},
        color_discrete_map={"Eligible": "#2E8B57", "Not eligible": "#D95F59"},
    )
    chart.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1])
    st.plotly_chart(chart, width="stretch", key=key)


def render_binned_approval_chart(
    data: pd.DataFrame,
    column: str,
    bins: list[int],
    labels: list[str],
    title: str,
    key: str,
) -> None:
    """Show loan outcomes by the notebook's income or loan-amount ranges."""
    values = pd.to_numeric(data[column], errors="coerce")
    status = data["Loan_Status"].astype(str).str.strip().str.upper()
    chart_data = pd.DataFrame(
        {
            "Range": pd.cut(values, bins=bins, labels=labels, include_lowest=True),
            "Outcome": status.map({"Y": "Eligible", "N": "Not eligible"}),
        }
    ).dropna()
    if chart_data.empty:
        st.info(f"No values are available for {title.lower()}.")
        return

    proportions = pd.crosstab(
        chart_data["Range"], chart_data["Outcome"], normalize="index"
    ).reindex(index=labels, columns=["Eligible", "Not eligible"], fill_value=0)
    chart = px.bar(
        proportions,
        x=proportions.index,
        y=proportions.columns,
        barmode="stack",
        title=title,
        labels={"x": "Range", "y": "Share of applications"},
        color_discrete_map={"Eligible": "#2E8B57", "Not eligible": "#D95F59"},
    )
    chart.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1])
    st.plotly_chart(chart, width="stretch", key=key)


def render_notebook_graphs(
    data: pd.DataFrame, test_data: pd.DataFrame, model: Pipeline
) -> None:
    """Render the exploratory and model charts from the analysis notebook."""
    applicant_tab, approval_tab, engineered_tab, model_tab = st.tabs(
        [
            "Applicant distributions",
            "Approval patterns",
            "Feature engineering",
            "Model importance",
        ]
    )

    with applicant_tab:
        left, right = st.columns(2)
        with left:
            render_histogram(
                data,
                "ApplicantIncome",
                "Applicant income distribution",
                "applicant-income-distribution",
                density=True,
            )
            render_box_plot(
                data,
                "ApplicantIncome",
                "Applicant income by education",
                "applicant-income-by-education",
                group_by="Education",
            )
            render_histogram(
                data,
                "CoapplicantIncome",
                "Co-applicant income distribution",
                "coapplicant-income-distribution",
                density=True,
            )
            render_box_plot(
                data,
                "CoapplicantIncome",
                "Co-applicant income range",
                "coapplicant-income-box",
            )
            render_histogram(
                data,
                "LoanAmount",
                "Loan amount distribution",
                "loan-amount-distribution",
                density=True,
            )
        with right:
            render_box_plot(
                data,
                "ApplicantIncome",
                "Applicant income range",
                "applicant-income-box",
            )
            render_box_plot(
                data,
                "LoanAmount",
                "Loan amount range",
                "loan-amount-box",
            )
            render_histogram(
                data,
                "Loan_Amount_Term",
                "Loan term distribution",
                "loan-term-distribution",
                density=True,
            )
            render_box_plot(
                data,
                "Loan_Amount_Term",
                "Loan term range",
                "loan-term-box",
            )

    with approval_tab:
        categorical_features = [
            "Gender",
            "Married",
            "Dependents",
            "Education",
            "Self_Employed",
            "Credit_History",
            "Property_Area",
        ]
        for start in range(0, len(categorical_features), 2):
            columns = st.columns(2)
            for column, feature in zip(
                columns, categorical_features[start : start + 2]
            ):
                with column:
                    render_approval_chart(data, feature, f"approval-by-{feature}")

        st.subheader("Approval by income and loan amount ranges")
        data_with_total_income = data.copy()
        data_with_total_income["TotalIncome"] = (
            pd.to_numeric(data["ApplicantIncome"], errors="coerce")
            + pd.to_numeric(data["CoapplicantIncome"], errors="coerce")
        )
        range_charts = [
            (
                "ApplicantIncome",
                [0, 2500, 4000, 6000, 81000],
                ["Low", "Average", "High", "Very high"],
                "Approval by applicant income",
                "approval-by-applicant-income",
            ),
            (
                "CoapplicantIncome",
                [0, 1000, 3000, 42000],
                ["Low", "Average", "High"],
                "Approval by co-applicant income",
                "approval-by-coapplicant-income",
            ),
            (
                "TotalIncome",
                [0, 2500, 4000, 6000, 81000],
                ["Low", "Average", "High", "Very high"],
                "Approval by combined household income",
                "approval-by-total-income",
            ),
            (
                "LoanAmount",
                [0, 100, 200, 700],
                ["Low", "Average", "High"],
                "Approval by loan amount",
                "approval-by-loan-amount",
            ),
        ]
        for start in range(0, len(range_charts), 2):
            columns = st.columns(2)
            for column, (field, bins, labels, title, key) in zip(
                columns, range_charts[start : start + 2]
            ):
                with column:
                    render_binned_approval_chart(
                        data_with_total_income, field, bins, labels, title, key
                    )

        correlation = data.select_dtypes(include="number").corr()
        if not correlation.empty:
            correlation_chart = px.imshow(
                correlation,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="BuPu",
                zmin=-1,
                zmax=1,
                title="Numeric feature correlation heatmap",
            )
            st.plotly_chart(
                correlation_chart,
                width="stretch",
                key="numeric-feature-correlation",
            )

    with engineered_tab:
        st.caption(
            "Log charts use each dataset's own positive values. The notebook's "
            "test-data log expression references training data; this dashboard "
            "uses the test data itself."
        )
        train_income = pd.to_numeric(data["ApplicantIncome"], errors="coerce")
        train_coapplicant_income = pd.to_numeric(
            data["CoapplicantIncome"], errors="coerce"
        )
        train_amount = pd.to_numeric(data["LoanAmount"], errors="coerce")
        train_term = pd.to_numeric(data["Loan_Amount_Term"], errors="coerce")
        test_income = pd.to_numeric(test_data["ApplicantIncome"], errors="coerce")
        test_coapplicant_income = pd.to_numeric(
            test_data["CoapplicantIncome"], errors="coerce"
        )
        test_amount = pd.to_numeric(test_data["LoanAmount"], errors="coerce")
        test_term = pd.to_numeric(test_data["Loan_Amount_Term"], errors="coerce")
        train_features = pd.DataFrame(
            {
                "LoanAmount": train_amount,
                "TotalIncome": train_income + train_coapplicant_income,
                "EMI": train_amount / train_term.where(train_term > 0),
            }
        )
        test_features = pd.DataFrame(
            {
                "LoanAmount": test_amount,
                "TotalIncome": test_income + test_coapplicant_income,
                "EMI": test_amount / test_term.where(test_term > 0),
            }
        )

        st.subheader("Loan amount: training and test data")
        loan_amount_charts = [
            (
                "Training loan amount distribution",
                train_features,
                "train-loan-amount-density",
                False,
                True,
            ),
            (
                "Training loan amount histogram",
                train_features,
                "train-loan-amount-hist",
                False,
                False,
            ),
            (
                "Training log loan amount histogram",
                train_features,
                "train-log-loan-amount-hist",
                True,
                False,
            ),
            (
                "Training log loan amount distribution",
                train_features,
                "train-log-loan-amount-density",
                True,
                True,
            ),
            (
                "Test log loan amount histogram",
                test_features,
                "test-log-loan-amount-hist",
                True,
                False,
            ),
            (
                "Test log loan amount distribution",
                test_features,
                "test-log-loan-amount-density",
                True,
                True,
            ),
        ]
        for start in range(0, len(loan_amount_charts), 2):
            columns = st.columns(2)
            for column, (title, frame, key, log_scale, density) in zip(
                columns, loan_amount_charts[start : start + 2]
            ):
                with column:
                    render_histogram(
                        frame,
                        "LoanAmount",
                        title,
                        key,
                        density=density,
                        log_scale=log_scale,
                    )

        st.subheader("Combined income: training and test data")
        income_charts = [
            (
                "Training combined income distribution",
                train_features,
                "train-total-income",
                False,
            ),
            (
                "Training log combined income",
                train_features,
                "train-log-total-income",
                True,
            ),
            (
                "Test combined income distribution",
                test_features,
                "test-total-income",
                False,
            ),
            (
                "Test log combined income",
                test_features,
                "test-log-total-income",
                True,
            ),
        ]
        for start in range(0, len(income_charts), 2):
            columns = st.columns(2)
            for column, (title, frame, key, log_scale) in zip(
                columns, income_charts[start : start + 2]
            ):
                with column:
                    render_histogram(
                        frame,
                        "TotalIncome",
                        title,
                        key,
                        density=True,
                        log_scale=log_scale,
                    )

        st.subheader("Estimated monthly installment (EMI)")
        columns = st.columns(2)
        for column, frame, title, key in [
            (
                columns[0],
                train_features,
                "Training EMI distribution",
                "train-emi",
            ),
            (columns[1], test_features, "Test EMI distribution", "test-emi"),
        ]:
            with column:
                render_histogram(frame, "EMI", title, key, density=True)

    with model_tab:
        preprocessing = model.named_steps["preprocessing"]
        classifier = model.named_steps["classifier"]
        importance = pd.DataFrame(
            {
                "Feature": preprocessing.get_feature_names_out(),
                "Importance": classifier.feature_importances_,
            }
        ).sort_values("Importance", ascending=True)
        importance_chart = px.bar(
            importance,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Random forest feature importance",
        )
        st.plotly_chart(
            importance_chart, width="stretch", key="feature-importance"
        )


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
        test_data = load_test_data(TEST_DATA_PATH)
    except (FileNotFoundError, ValueError) as error:
        st.error(f"Could not load the loan prediction app: {error}")
        st.stop()

    overview_tab, graphs_tab, prediction_tab, data_tab = st.tabs(
        [
            "Overview",
            "All notebook graphs",
            "Predict eligibility",
            "Explore all data",
        ]
    )

    with overview_tab:
        st.header("Loan application overview")
        st.caption("Charts and summary statistics are based on the training sample.")
        render_overview(data)

    with graphs_tab:
        st.header("Graphs from the project notebook")
        st.caption(
            "Explore applicant distributions, approval comparisons, "
            "feature-engineering charts, and random-forest feature importance."
        )
        render_notebook_graphs(data, test_data, model)

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
        st.header("Complete datasets")
        training_data_tab, test_data_tab = st.tabs(["Training data", "Test data"])
        with training_data_tab:
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
        with test_data_tab:
            st.write(
                f"All {len(test_data):,} test records and their original columns. "
                "This sample does not include loan outcomes."
            )
            st.download_button(
                "Download test data as CSV",
                data=test_data.to_csv(index=False).encode("utf-8"),
                file_name="home_loan_test_data.csv",
                mime="text/csv",
            )
            st.dataframe(test_data, hide_index=True)

    st.caption(
        "For demonstration only. This prediction is not a lending decision, "
        "financial advice, or a guarantee of loan approval."
    )


if __name__ == "__main__":
    main()
