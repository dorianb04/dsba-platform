import logging
from datetime import datetime

import pandas as pd
import requests
import xgboost as xgb
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split

# Import necessary components from other dsba modules
from dsba.data_ingestion import load_csv_from_path, load_csv_from_url
from dsba.model_evaluation import ClassifierEvaluationResult, evaluate_classifier
from dsba.model_registry import ClassifierMetadata
from dsba.preprocessing import preprocess_dataframe, split_features_and_target

logger = logging.getLogger(__name__)

# --- Helper Functions ---


def _load_data(data_source: str) -> pd.DataFrame:
    logger.info(f"Loading data from source: {data_source}")
    try:
        if data_source.startswith(("http://", "https://")):
            df = load_csv_from_url(data_source)
        else:
            df = load_csv_from_path(data_source)

        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        if df.empty:
            logger.warning(f"Loaded dataframe from {data_source} is empty.")
        return df
    except FileNotFoundError as e:
        logger.error(f"Data file not found via data_ingestion: {e}")
        raise e
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error loading data via data_ingestion: {e}")
        raise ConnectionError(f"Could not connect/read URL {data_source}") from e
    except Exception as e:
        logger.error(f"Failed to load data via data_ingestion from {data_source}: {e}")
        raise ValueError(f"Could not load/parse data from {data_source}") from e


def _prepare_data_for_training(
    df: pd.DataFrame, target_column: str, test_size: float
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Preprocesses, checks, splits data,
    and returns train features, train target, test df.

    """
    logger.info("Preparing data for training...")

    if target_column not in df.columns:
        msg = f"Target column '{target_column}' not found in the data."
        logger.error(msg)
        raise ValueError(msg)

    df_cleaned = df.dropna(subset=[target_column]).copy()
    if df_cleaned.empty:
        msg = "DataFrame empty after dropping rows with NaN target."
        logger.error(msg)
        raise ValueError(msg)

    logger.info("Preprocessing features...")
    df_processed = preprocess_dataframe(df_cleaned)

    logger.info(f"Splitting data (test_size={test_size})...")
    try:
        stratify_col = (
            df_processed[target_column]
            if df_processed[target_column].nunique() > 1
            else None
        )
        train_df, test_df = train_test_split(
            df_processed, test_size=test_size, random_state=42, stratify=stratify_col
        )
    except Exception as stratify_err:
        logger.warning(f"Stratification failed ({stratify_err}), splitting without.")
        train_df, test_df = train_test_split(
            df_processed, test_size=test_size, random_state=42
        )

    if train_df.empty or test_df.empty:
        msg = "Train or test DataFrame is empty after splitting."
        logger.error(msg)
        raise ValueError(msg)

    logger.info("Separating features and target for training set...")
    x_train, y_train = split_features_and_target(train_df, target_column)

    logger.info(
        f"Data preparation complete. Train features shape: {x_train.shape}, "
        f"Test df shape: {test_df.shape}"
    )
    return x_train, y_train, test_df


def _train_model(x_train: pd.DataFrame, y_train: pd.Series) -> ClassifierMixin:
    """Instantiates and trains the XGBoost model."""
    logger.info("Training XGBoost classifier...")
    try:
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            random_state=42,
        )
        model.fit(x_train, y_train)
        logger.info("Model training completed.")
        return model
    except Exception as e:
        logger.exception(f"Error during model training execution: {e}")
        raise RuntimeError("Model training failed") from e


def _evaluate_model(
    model: ClassifierMixin, test_df: pd.DataFrame, target_column: str
) -> dict[str, float | str]:
    """Evaluates the model and returns performance metrics."""
    logger.info("Evaluating model on the test set...")
    performance_metrics = {}
    try:
        if target_column not in test_df.columns:
            raise ValueError(
                f"Target column '{target_column}' missing in test data for eval."
            )

        evaluation_results: ClassifierEvaluationResult = evaluate_classifier(
            model, target_column, test_df
        )
        logger.info(
            f"Eval Metrics: Acc={evaluation_results.accuracy:.4f}, "
            f"F1={evaluation_results.f1_score:.4f}"
        )
        performance_metrics = {
            "accuracy": round(evaluation_results.accuracy, 5),
            "precision": round(evaluation_results.precision, 5),
            "recall": round(evaluation_results.recall, 5),
            "f1_score": round(evaluation_results.f1_score, 5),
        }
        return performance_metrics
    except Exception as e:
        logger.exception(f"Error during model evaluation: {e}")
        return {"error": f"Evaluation failed: {e!s}"}


def _create_metadata(
    model_id: str,
    target_column: str,
    data_source: str,
    model: ClassifierMixin,
    performance_metrics: dict[str, float | str],
) -> ClassifierMetadata:
    """Creates the metadata object for the trained model."""
    logger.info("Creating model metadata...")
    hyperparams = {}
    try:
        raw_params = model.get_params()
        for k, v in raw_params.items():
            if hasattr(v, "tolist"):
                hyperparams[k] = v.tolist()
            elif isinstance(v, pd.Timestamp | pd.Timedelta):
                hyperparams[k] = str(v)
            elif hasattr(v, "dtype"):
                hyperparams[k] = str(v)
            else:
                hyperparams[k] = v
    except Exception as e:
        logger.warning(f"Could not serialize all hyperparameters: {e}")
        hyperparams = {"error": "Could not serialize parameters"}

    metadata = ClassifierMetadata(
        id=model_id,
        created_at=datetime.now().isoformat(),
        algorithm=getattr(model, "_estimator_type", "Unknown")
        + ":"
        + model.__class__.__name__,
        target_column=target_column,
        hyperparameters=hyperparams,
        description=f"Model trained on data from {data_source}",
        performance_metrics=performance_metrics,
    )
    return metadata


# --- Main Orchestration Function ---


def run_training_pipeline(
    data_source: str, target_column: str, model_id: str, test_size: float = 0.2
) -> tuple[ClassifierMixin, ClassifierMetadata]:
    """
    Orchestrates the full training pipeline: load, prepare, train, evaluate, metadata.
    """
    logger.info(f"--- Starting Training Pipeline for model_id: {model_id} ---")
    raw_df = _load_data(data_source)
    x_train, y_train, test_df = _prepare_data_for_training(
        raw_df, target_column, test_size
    )
    # Pass renamed variable
    model = _train_model(x_train, y_train)
    performance_metrics = _evaluate_model(model, test_df, target_column)
    metadata = _create_metadata(
        model_id, target_column, data_source, model, performance_metrics
    )
    logger.info(
        f"--- Training Pipeline finished successfully for model_id: {model_id} ---"
    )
    return model, metadata
