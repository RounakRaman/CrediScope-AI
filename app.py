from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from credit_risk_core import (
    DEFAULT_TARGET,
    build_input_defaults,
    clean_dataset_for_download,
    cluster_borrowers,
    dataframe_to_csv_bytes,
    download_kaggle_dataset,
    feature_importance_table,
    metrics_at_threshold,
    model_to_bytes,
    one_hot_correlations,
    predict_applicants,
    read_uploaded_csv,
    train_models,
    validate_binary_target,
)

st.set_page_config(
    page_title="CrediScope AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False, ttl=3600)
def load_kaggle_data() -> pd.DataFrame:
    return download_kaggle_dataset()


@st.cache_resource(show_spinner=False)
def cached_train_models(
    df: pd.DataFrame,
    target: str,
    test_size: float,
    random_state: int,
    class_weight: str | None,
    n_estimators: int,
):
    return train_models(
        df=df,
        target=target,
        test_size=test_size,
        random_state=random_state,
        class_weight=class_weight,
        n_estimators=n_estimators,
    )


def format_metric(value: float) -> str:
    return f"{value:.3f}"


def numeric_input_widget(column: str, series: pd.Series, default: float) -> float:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return st.number_input(column, value=float(default))

    minimum = float(clean.min())
    maximum = float(clean.max())
    spread = maximum - minimum
    lower = minimum - spread * 0.25 if spread > 0 else minimum - 1.0
    upper = maximum + spread * 0.25 if spread > 0 else maximum + 1.0
    step = max(spread / 100.0, 0.01)

    is_integer = pd.api.types.is_integer_dtype(series.dropna())
    if is_integer:
        return float(
            st.number_input(
                column,
                min_value=int(math.floor(lower)),
                max_value=int(math.ceil(upper)),
                value=int(round(default)),
                step=max(1, int(round(step))),
            )
        )
    return float(
        st.number_input(
            column,
            min_value=float(lower),
            max_value=float(upper),
            value=float(default),
            step=float(step),
            format="%.4f",
        )
    )


def risk_gauge(probability: float, threshold: float) -> go.Figure:
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "valueformat": ".1f"},
            title={"text": "Estimated default probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"thickness": 0.28},
                "steps": [
                    {"range": [0, 25]},
                    {"range": [25, 50]},
                    {"range": [50, 75]},
                    {"range": [75, 100]},
                ],
                "threshold": {
                    "line": {"width": 4},
                    "thickness": 0.85,
                    "value": threshold * 100,
                },
            },
        )
    )
    figure.update_layout(height=330, margin=dict(l=20, r=20, t=60, b=20))
    return figure


st.title("CrediScope AI")
st.caption(
    "Interactive credit-risk analytics, model benchmarking, borrower segmentation, "
    "and individual or batch applicant scoring."
)

with st.sidebar:
    st.header("Control Center")
    data_source = st.radio(
        "Dataset source",
        ["Kaggle dataset", "Upload CSV"],
        help="Use the original public dataset or upload a compatible CSV.",
    )

    uploaded_file = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload credit-risk CSV", type=["csv"])

    st.divider()
    st.subheader("Training settings")
    test_size = st.slider("Test set size", 0.10, 0.40, 0.20, 0.05)
    random_state = st.number_input("Random seed", min_value=0, value=42, step=1)
    balance_classes = st.toggle(
        "Balance target classes",
        value=False,
        help="Uses class_weight='balanced' for both classifiers.",
    )
    n_estimators = st.slider("Random-forest trees", 100, 800, 300, 50)
    decision_threshold = st.slider(
        "Decision threshold",
        0.10,
        0.90,
        0.50,
        0.01,
        help="Lower values increase sensitivity to risky applicants; higher values reduce false alerts.",
    )

try:
    if data_source == "Kaggle dataset":
        with st.spinner("Loading the public credit-risk dataset..."):
            data = load_kaggle_data()
        source_label = "Kaggle: laotse/credit-risk-dataset"
    elif uploaded_file is not None:
        data = read_uploaded_csv(uploaded_file)
        source_label = uploaded_file.name
    else:
        st.info("Upload a CSV file to begin analysis.")
        st.stop()
except Exception as exc:
    st.error(f"The dataset could not be loaded: {exc}")
    st.info("Switch to Upload CSV in the sidebar and provide a local dataset.")
    st.stop()

if data.empty:
    st.error("The selected dataset contains no rows.")
    st.stop()

with st.sidebar:
    target_default_index = data.columns.get_loc(DEFAULT_TARGET) if DEFAULT_TARGET in data.columns else 0
    target = st.selectbox("Target column", data.columns.tolist(), index=target_default_index)

try:
    validate_binary_target(data, target)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

class_weight = "balanced" if balance_classes else None
with st.spinner("Training and evaluating credit-risk models..."):
    try:
        training = cached_train_models(
            data,
            target,
            float(test_size),
            int(random_state),
            class_weight,
            int(n_estimators),
        )
    except Exception as exc:
        st.error(f"Model training failed: {exc}")
        st.stop()

best_model_name = training.metrics.iloc[0]["Model"]

summary_columns = st.columns(5)
summary_columns[0].metric("Rows", f"{len(data):,}")
summary_columns[1].metric("Predictors", f"{data.shape[1] - 1:,}")
summary_columns[2].metric("Missing cells", f"{int(data.isna().sum().sum()):,}")
summary_columns[3].metric("Positive class", str(training.positive_label))
summary_columns[4].metric("Best ROC AUC", format_metric(training.metrics.iloc[0]["ROC AUC"]))

st.caption(f"Source: {source_label} · Best validation model: {best_model_name}")

overview_tab, modeling_tab, scoring_tab, segments_tab, explorer_tab = st.tabs(
    [
        "Portfolio Overview",
        "Model Lab",
        "Applicant Scoring",
        "Borrower Segments",
        "Data Explorer",
    ]
)

with overview_tab:
    st.subheader("Dataset health and portfolio structure")
    target_counts = data[target].value_counts(dropna=False).rename_axis("Class").reset_index(name="Borrowers")
    target_counts["Share"] = target_counts["Borrowers"] / target_counts["Borrowers"].sum()

    left, right = st.columns(2)
    with left:
        fig_target = px.bar(
            target_counts,
            x="Class",
            y="Borrowers",
            text_auto=True,
            title="Target-class distribution",
        )
        fig_target.update_layout(xaxis_type="category")
        st.plotly_chart(fig_target, width="stretch")
    with right:
        missing = data.isna().sum().sort_values(ascending=False)
        missing = missing[missing > 0].rename("Missing values").reset_index(names="Feature")
        if missing.empty:
            st.success("No missing values were detected.")
        else:
            fig_missing = px.bar(
                missing,
                x="Missing values",
                y="Feature",
                orientation="h",
                title="Missing values by feature",
            )
            fig_missing.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_missing, width="stretch")

    numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
    chart_numeric_columns = [column for column in numeric_columns if column != target]
    if chart_numeric_columns:
        selected_numeric = st.selectbox("Explore a numeric feature", chart_numeric_columns)
        histogram = px.histogram(
            data,
            x=selected_numeric,
            color=data[target].astype(str),
            barmode="overlay",
            marginal="box",
            title=f"{selected_numeric} by {target}",
        )
        histogram.update_traces(opacity=0.65)
        st.plotly_chart(histogram, width="stretch")

    st.subheader("Strongest feature relationships with the target")
    try:
        correlation_table = one_hot_correlations(data, target, top_n=25)
        correlation_figure = px.bar(
            correlation_table.sort_values("Correlation"),
            x="Correlation",
            y="Feature",
            orientation="h",
            title="Absolute strongest one-hot feature correlations",
        )
        st.plotly_chart(correlation_figure, width="stretch")
    except Exception as exc:
        st.warning(f"Correlation analysis is unavailable: {exc}")

with modeling_tab:
    st.subheader("Model benchmark")
    metrics_display = training.metrics.copy()
    metric_columns = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
    metrics_display[metric_columns] = metrics_display[metric_columns].round(4)
    st.dataframe(metrics_display, hide_index=True, width="stretch")

    roc_figure = go.Figure()
    for model_name, curve in training.curves.items():
        auc_value = training.metrics.loc[training.metrics["Model"] == model_name, "ROC AUC"].iloc[0]
        roc_figure.add_trace(
            go.Scatter(
                x=curve["fpr"],
                y=curve["tpr"],
                mode="lines",
                name=f"{model_name} (AUC {auc_value:.3f})",
            )
        )
    roc_figure.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random baseline", line={"dash": "dash"})
    )
    roc_figure.update_layout(
        title="ROC curves",
        xaxis_title="False-positive rate",
        yaxis_title="True-positive rate",
        height=450,
    )
    st.plotly_chart(roc_figure, width="stretch")

    selected_model_name = st.selectbox("Inspect model", list(training.models.keys()), index=list(training.models.keys()).index(best_model_name))
    selected_model = training.models[selected_model_name]
    threshold_metrics = metrics_at_threshold(
        selected_model,
        training.X_test,
        training.y_test,
        decision_threshold,
    )

    threshold_columns = st.columns(5)
    for column, metric_name in zip(
        threshold_columns,
        ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
    ):
        column.metric(metric_name, format_metric(threshold_metrics[metric_name]))

    matrix = threshold_metrics["Confusion Matrix"]
    confusion_figure = px.imshow(
        matrix,
        text_auto=True,
        labels={"x": "Predicted class", "y": "Actual class", "color": "Borrowers"},
        x=["Lower risk", "Higher risk"],
        y=["Lower risk", "Higher risk"],
        title=f"Confusion matrix at threshold {decision_threshold:.2f}",
        aspect="auto",
    )

    importance = feature_importance_table(selected_model, top_n=20)
    importance_figure = px.bar(
        importance.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        title=f"Top drivers: {selected_model_name}",
    )

    matrix_col, importance_col = st.columns(2)
    matrix_col.plotly_chart(confusion_figure, width="stretch")
    importance_col.plotly_chart(importance_figure, width="stretch")

    model_download_col, metrics_download_col = st.columns(2)
    model_download_col.download_button(
        "Download trained model",
        data=model_to_bytes(selected_model),
        file_name=f"{selected_model_name.lower().replace(' ', '_')}_pipeline.joblib",
        mime="application/octet-stream",
        width="stretch",
    )
    metrics_download_col.download_button(
        "Download model metrics",
        data=dataframe_to_csv_bytes(metrics_display),
        file_name="credit_risk_model_metrics.csv",
        mime="text/csv",
        width="stretch",
    )

with scoring_tab:
    st.subheader("Score one applicant")
    scoring_model_name = st.selectbox(
        "Scoring model",
        list(training.models.keys()),
        index=list(training.models.keys()).index(best_model_name),
        key="scoring_model",
    )
    scoring_model = training.models[scoring_model_name]
    predictors = data.drop(columns=[target])
    defaults = build_input_defaults(predictors)

    applicant_values: dict[str, Any] = {}
    with st.form("applicant_form"):
        form_columns = st.columns(3)
        for index, column in enumerate(predictors.columns):
            with form_columns[index % 3]:
                if pd.api.types.is_numeric_dtype(predictors[column]):
                    applicant_values[column] = numeric_input_widget(
                        column,
                        predictors[column],
                        float(defaults[column]),
                    )
                else:
                    options = predictors[column].dropna().astype(str).unique().tolist()
                    options = sorted(options)
                    default_as_string = str(defaults[column])
                    default_index = options.index(default_as_string) if default_as_string in options else 0
                    applicant_values[column] = st.selectbox(column, options, index=default_index)
        score_applicant = st.form_submit_button("Calculate risk score", width="stretch")

    if score_applicant:
        applicant = pd.DataFrame([applicant_values])
        for column in predictors.select_dtypes(exclude=np.number).columns:
            applicant[column] = applicant[column].astype(predictors[column].dtype)
        scored = predict_applicants(scoring_model, applicant, decision_threshold)
        probability = float(scored.loc[0, "default_probability"])

        gauge_col, decision_col = st.columns([2, 1])
        with gauge_col:
            st.plotly_chart(risk_gauge(probability, decision_threshold), width="stretch")
        with decision_col:
            st.metric("Risk band", scored.loc[0, "risk_band"])
            st.metric("Decision", scored.loc[0, "risk_decision"])
            st.metric("Threshold", f"{decision_threshold:.0%}")
            st.caption(
                "This is a statistical estimate from the selected dataset and is not a lending decision or financial advice."
            )

    st.divider()
    st.subheader("Batch scoring")
    template = pd.DataFrame([defaults])
    st.download_button(
        "Download batch template",
        data=dataframe_to_csv_bytes(template),
        file_name="credit_risk_batch_template.csv",
        mime="text/csv",
    )
    batch_file = st.file_uploader("Upload completed applicant batch", type=["csv"], key="batch_file")
    if batch_file is not None:
        try:
            batch = pd.read_csv(batch_file)
            missing_columns = [column for column in predictors.columns if column not in batch.columns]
            if missing_columns:
                st.error("Missing predictor columns: " + ", ".join(missing_columns))
            else:
                batch = batch[predictors.columns]
                batch_result = predict_applicants(scoring_model, batch, decision_threshold)
                st.dataframe(batch_result, hide_index=True, width="stretch")
                st.download_button(
                    "Download scored applicants",
                    data=dataframe_to_csv_bytes(batch_result),
                    file_name="scored_credit_applicants.csv",
                    mime="text/csv",
                    width="stretch",
                )
        except Exception as exc:
            st.error(f"Batch scoring failed: {exc}")

with segments_tab:
    st.subheader("Unsupervised borrower segmentation")
    numeric_predictors = data.drop(columns=[target]).select_dtypes(include=np.number).columns.tolist()
    preferred_features = [
        column
        for column in ["loan_percent_income", "person_income", "loan_int_rate"]
        if column in numeric_predictors
    ]
    if len(preferred_features) < 2:
        preferred_features = numeric_predictors[: min(3, len(numeric_predictors))]

    if len(numeric_predictors) < 2:
        st.warning("At least two numeric predictors are required for clustering.")
    else:
        cluster_features = st.multiselect(
            "Clustering features",
            numeric_predictors,
            default=preferred_features,
            max_selections=6,
        )
        cluster_count = st.slider("Number of clusters", 2, 7, 3)

        if len(cluster_features) >= 2:
            try:
                cluster_result, cluster_profile, silhouette, cluster_plot = cluster_borrowers(
                    data,
                    cluster_features,
                    cluster_count,
                    int(random_state),
                )
                st.metric("Silhouette score", f"{silhouette:.3f}")

                segment_figure = px.scatter(
                    cluster_plot,
                    x="PC1",
                    y="PC2",
                    color="Cluster",
                    hover_data=cluster_features,
                    title="PCA projection of K-Means borrower segments",
                    opacity=0.65,
                )
                st.plotly_chart(segment_figure, width="stretch")

                st.subheader("Segment profiles")
                st.dataframe(cluster_profile.reset_index(), hide_index=True, width="stretch")

                downloadable_clusters = data.copy()
                downloadable_clusters["borrower_cluster"] = cluster_result["Cluster"]
                st.download_button(
                    "Download segmented portfolio",
                    data=dataframe_to_csv_bytes(downloadable_clusters),
                    file_name="segmented_credit_portfolio.csv",
                    mime="text/csv",
                    width="stretch",
                )
            except Exception as exc:
                st.warning(f"Clustering could not be completed: {exc}")
        else:
            st.info("Select at least two numeric features.")

with explorer_tab:
    st.subheader("Interactive data explorer")
    filtered_data = data.copy()
    categorical_columns = data.select_dtypes(exclude=np.number).columns.tolist()
    filter_columns = st.multiselect(
        "Add categorical filters",
        categorical_columns,
        default=categorical_columns[: min(2, len(categorical_columns))],
    )

    if filter_columns:
        filter_layout = st.columns(min(3, len(filter_columns)))
        for index, column in enumerate(filter_columns):
            options = sorted(data[column].dropna().astype(str).unique().tolist())
            selected = filter_layout[index % len(filter_layout)].multiselect(
                column,
                options,
                default=options,
                key=f"filter_{column}",
            )
            filtered_data = filtered_data[filtered_data[column].astype(str).isin(selected)]

    st.caption(f"Displaying {len(filtered_data):,} of {len(data):,} rows")
    st.dataframe(filtered_data, hide_index=True, width="stretch", height=520)

    cleaned = clean_dataset_for_download(filtered_data)
    raw_col, clean_col = st.columns(2)
    raw_col.download_button(
        "Download filtered raw data",
        data=dataframe_to_csv_bytes(filtered_data),
        file_name="filtered_credit_risk_data.csv",
        mime="text/csv",
        width="stretch",
    )
    clean_col.download_button(
        "Download median/mode cleaned data",
        data=dataframe_to_csv_bytes(cleaned),
        file_name="cleaned_credit_risk_data.csv",
        mime="text/csv",
        width="stretch",
    )

st.divider()
st.caption(
    "CrediScope AI is an educational analytics prototype. Validate fairness, calibration, governance, "
    "privacy, and regulatory requirements before any real-world credit use."
)
