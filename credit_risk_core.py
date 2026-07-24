from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import kagglehub
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATASET_HANDLE = "laotse/credit-risk-dataset"
DATASET_FILENAME = "credit_risk_dataset.csv"
DEFAULT_TARGET = "loan_status"


@dataclass
class TrainingResult:
    models: dict[str, Pipeline]
    metrics: pd.DataFrame
    curves: dict[str, dict[str, np.ndarray]]
    confusion_matrices: dict[str, np.ndarray]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    target_mapping: dict[Any, int]
    positive_label: Any


def download_kaggle_dataset() -> pd.DataFrame:
    """Download the public Kaggle dataset and return it as a DataFrame."""
    dataset_path = Path(kagglehub.dataset_download(DATASET_HANDLE))
    csv_path = dataset_path / DATASET_FILENAME
    if not csv_path.exists():
        csv_files = list(dataset_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV file was found in the downloaded dataset at {dataset_path}."
            )
        csv_path = csv_files[0]
    return pd.read_csv(csv_path)


def read_uploaded_csv(uploaded_file: Any) -> pd.DataFrame:
    """Read a Streamlit UploadedFile or file-like object safely."""
    return pd.read_csv(uploaded_file)


def validate_binary_target(df: pd.DataFrame, target: str) -> None:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' is missing from the dataset.")
    non_null = df[target].dropna()
    unique_values = non_null.unique().tolist()
    if len(unique_values) != 2:
        raise ValueError(
            f"'{target}' must contain exactly two non-null classes; found {len(unique_values)}."
        )
    if df.drop(columns=[target]).shape[1] == 0:
        raise ValueError("At least one predictor column is required.")


def infer_positive_label(values: pd.Series) -> Any:
    unique_values = values.dropna().unique().tolist()
    for preferred in (1, True, "1", "default", "bad", "yes", "positive"):
        if preferred in unique_values:
            return preferred
    try:
        return sorted(unique_values)[-1]
    except TypeError:
        return unique_values[-1]


def encode_binary_target(values: pd.Series) -> tuple[pd.Series, dict[Any, int], Any]:
    positive_label = infer_positive_label(values)
    negative_label = next(value for value in values.dropna().unique() if value != positive_label)
    mapping = {negative_label: 0, positive_label: 1}
    encoded = values.map(mapping)
    if encoded.isna().any():
        raise ValueError("The target contains missing or unmapped values.")
    return encoded.astype(int), mapping, positive_label


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_columns = X.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        transformers.append(("num", numeric_pipeline, numeric_columns))
    if categorical_columns:
        transformers.append(("cat", categorical_pipeline, categorical_columns))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor, numeric_columns, categorical_columns


def train_models(
    df: pd.DataFrame,
    target: str = DEFAULT_TARGET,
    test_size: float = 0.2,
    random_state: int = 42,
    class_weight: str | None = None,
    n_estimators: int = 300,
) -> TrainingResult:
    validate_binary_target(df, target)

    model_df = df.dropna(subset=[target]).copy()
    X = model_df.drop(columns=[target])
    y, mapping, positive_label = encode_binary_target(model_df[target])

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    logistic_preprocessor, _, _ = build_preprocessor(X_train)
    forest_preprocessor, _, _ = build_preprocessor(X_train)

    models: dict[str, Pipeline] = {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", logistic_preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight=class_weight,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", forest_preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=n_estimators,
                        random_state=random_state,
                        class_weight=class_weight,
                        n_jobs=-1,
                        min_samples_leaf=2,
                    ),
                ),
            ]
        ),
    }

    metric_rows: list[dict[str, float | str]] = []
    curves: dict[str, dict[str, np.ndarray]] = {}
    confusion_matrices: dict[str, np.ndarray] = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        probability = model.predict_proba(X_test)[:, 1]
        prediction = (probability >= 0.5).astype(int)

        metric_rows.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, prediction),
                "Precision": precision_score(y_test, prediction, zero_division=0),
                "Recall": recall_score(y_test, prediction, zero_division=0),
                "F1 Score": f1_score(y_test, prediction, zero_division=0),
                "ROC AUC": roc_auc_score(y_test, probability),
            }
        )
        fpr, tpr, thresholds = roc_curve(y_test, probability)
        curves[model_name] = {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}
        confusion_matrices[model_name] = confusion_matrix(y_test, prediction)

    metrics = pd.DataFrame(metric_rows).sort_values("ROC AUC", ascending=False)
    return TrainingResult(
        models=models,
        metrics=metrics,
        curves=curves,
        confusion_matrices=confusion_matrices,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        target_mapping=mapping,
        positive_label=positive_label,
    )


def metrics_at_threshold(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> dict[str, Any]:
    probability = model.predict_proba(X_test)[:, 1]
    prediction = (probability >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_test, prediction),
        "Precision": precision_score(y_test, prediction, zero_division=0),
        "Recall": recall_score(y_test, prediction, zero_division=0),
        "F1 Score": f1_score(y_test, prediction, zero_division=0),
        "ROC AUC": roc_auc_score(y_test, probability),
        "Confusion Matrix": confusion_matrix(y_test, prediction),
    }


def transformed_feature_names(model: Pipeline) -> list[str]:
    preprocessor: ColumnTransformer = model.named_steps["preprocessor"]
    try:
        return preprocessor.get_feature_names_out().tolist()
    except AttributeError:
        names: list[str] = []
        for name, transformer, columns in preprocessor.transformers_:
            if name == "remainder" or transformer == "drop":
                continue
            if name == "num":
                names.extend([str(column) for column in columns])
            elif name == "cat":
                encoder = transformer.named_steps["encoder"]
                names.extend(encoder.get_feature_names_out(columns).tolist())
        return names


def feature_importance_table(model: Pipeline, top_n: int = 20) -> pd.DataFrame:
    estimator = model.named_steps["model"]
    names = transformed_feature_names(model)

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_[0])
    else:
        raise ValueError("The selected model does not expose feature importance values.")

    table = pd.DataFrame({"Feature": names, "Importance": values})
    table["Feature"] = (
        table["Feature"].str.replace("num__", "", regex=False).str.replace("cat__", "", regex=False)
    )
    return table.sort_values("Importance", ascending=False).head(top_n).reset_index(drop=True)


def one_hot_correlations(df: pd.DataFrame, target: str, top_n: int = 25) -> pd.DataFrame:
    working = df.dropna(subset=[target]).copy()
    y, _, _ = encode_binary_target(working[target])
    X = working.drop(columns=[target])

    numeric_columns = X.select_dtypes(include=np.number).columns
    categorical_columns = X.columns.difference(numeric_columns)
    X[numeric_columns] = X[numeric_columns].fillna(X[numeric_columns].median())
    for column in categorical_columns:
        mode = X[column].mode(dropna=True)
        X[column] = X[column].fillna(mode.iloc[0] if not mode.empty else "Unknown")

    encoded = pd.get_dummies(X, drop_first=True, dtype=float)
    encoded[target] = y.to_numpy()
    correlations = encoded.corr(numeric_only=True)[target].drop(target).dropna()
    result = correlations.reindex(correlations.abs().sort_values(ascending=False).index)
    result = result.head(top_n).rename("Correlation")
    result.index.name = "Feature"
    return result.reset_index()


def clean_dataset_for_download(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    numeric_columns = cleaned.select_dtypes(include=np.number).columns
    categorical_columns = cleaned.columns.difference(numeric_columns)
    cleaned[numeric_columns] = cleaned[numeric_columns].fillna(cleaned[numeric_columns].median())
    for column in categorical_columns:
        mode = cleaned[column].mode(dropna=True)
        cleaned[column] = cleaned[column].fillna(mode.iloc[0] if not mode.empty else "Unknown")
    return cleaned


def build_input_defaults(X: pd.DataFrame) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for column in X.columns:
        if pd.api.types.is_numeric_dtype(X[column]):
            defaults[column] = float(X[column].median()) if X[column].notna().any() else 0.0
        else:
            mode = X[column].mode(dropna=True)
            defaults[column] = mode.iloc[0] if not mode.empty else "Unknown"
    return defaults


def predict_applicants(
    model: Pipeline,
    applicants: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    probability = model.predict_proba(applicants)[:, 1]
    result = applicants.copy()
    result["default_probability"] = probability
    result["risk_decision"] = np.where(probability >= threshold, "Higher risk", "Lower risk")
    result["risk_band"] = pd.cut(
        probability,
        bins=[-np.inf, 0.25, 0.50, 0.75, np.inf],
        labels=["Low", "Moderate", "High", "Critical"],
    ).astype(str)
    return result


def cluster_borrowers(
    df: pd.DataFrame,
    features: list[str],
    n_clusters: int = 3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, float, pd.DataFrame]:
    if len(features) < 2:
        raise ValueError("Select at least two numeric features for clustering.")

    cluster_data = df[features].copy()
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    matrix = scaler.fit_transform(imputer.fit_transform(cluster_data))

    if len(cluster_data) <= n_clusters:
        raise ValueError("The dataset must contain more rows than the selected clusters.")

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = model.fit_predict(matrix)

    sample_size = min(5000, len(matrix))
    score = silhouette_score(
        matrix,
        labels,
        sample_size=sample_size if sample_size < len(matrix) else None,
        random_state=random_state,
    )

    result = cluster_data.copy()
    result["Cluster"] = labels
    profile = result.groupby("Cluster")[features].mean().round(3)
    profile["Borrowers"] = result.groupby("Cluster").size()

    pca = PCA(n_components=2, random_state=random_state)
    components = pca.fit_transform(matrix)
    plot_data = pd.DataFrame(
        {
            "PC1": components[:, 0],
            "PC2": components[:, 1],
            "Cluster": labels.astype(str),
        },
        index=result.index,
    )
    for feature in features:
        plot_data[feature] = result[feature]

    return result, profile, score, plot_data


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def model_to_bytes(model: Pipeline) -> bytes:
    buffer = BytesIO()
    joblib.dump(model, buffer)
    return buffer.getvalue()
