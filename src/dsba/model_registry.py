import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
from sklearn.base import BaseEstimator

# Import the config loading utility specifically for the path
from dsba.config import get_models_root_path

logger = logging.getLogger(__name__)


@dataclass
class ClassifierMetadata:
    id: str
    target_column: str
    algorithm: str = "Unknown"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    performance_metrics: dict[str, float] = field(default_factory=dict)


def save_model(model: BaseEstimator, metadata: ClassifierMetadata) -> None:
    """Saves the model and its metadata to the configured registry path."""
    # Get path via config utility and ensure it exists
    models_dir = _get_models_dir_and_ensure_exists()
    model_path = models_dir / f"{metadata.id}.pkl"
    model_metadata_path = models_dir / f"{metadata.id}.json"

    logger.info(f"Saving model '{metadata.id}' to {model_path}")
    try:
        joblib.dump(model, model_path)
        logger.info(
            f"Saving metadata for model '{metadata.id}' to {model_metadata_path}"
        )
        with open(model_metadata_path, "w") as f:
            json.dump(asdict(metadata), f, indent=4, default=str)
    except Exception as e:
        logger.exception(f"Failed to save model or metadata for '{metadata.id}': {e}")
        # Cleanup attempt
        model_path.unlink(missing_ok=True)
        model_metadata_path.unlink(missing_ok=True)
        raise  # Re-raise


def list_models_ids() -> list[str]:
    """Lists the IDs of models available in the registry."""
    models_dir = _get_models_dir_and_ensure_exists()  # Ensure dir exists before listing
    try:
        # List .pkl files and derive IDs
        model_files = [
            f for f in models_dir.iterdir() if f.is_file() and f.suffix == ".pkl"
        ]
        models_ids = [f.stem for f in model_files]
        logger.info(f"Found models in registry: {models_ids}")
        return models_ids
    except OSError as e:
        logger.exception(f"Error listing models in directory {models_dir}: {e}")
        return []


def load_model(model_id: str) -> BaseEstimator:
    """Loads a model artifact from the registry."""
    model_path = _get_model_path(model_id)  # Gets path using config
    if not model_path.exists():
        logger.error(
            f"Model artifact file not found for ID '{model_id}' at {model_path}"
        )
        raise FileNotFoundError(f"Model artifact file not found for ID '{model_id}'")
    logger.info(f"Loading model '{model_id}' from {model_path}")
    try:
        return joblib.load(model_path)
    except Exception as e:
        logger.exception(
            f"Failed to load model artifact '{model_id}' from {model_path}: {e}"
        )
        raise  # Re-raise


def load_model_metadata(model_id: str) -> ClassifierMetadata:
    """Loads model metadata from the registry."""
    # Gets path using config
    metadata_path = _get_model_metadata_path(model_id)
    if not metadata_path.exists():
        logger.error(f"Metadata file not found for ID '{model_id}' at {metadata_path}")
        raise FileNotFoundError(f"Metadata file not found for ID '{model_id}'")
    logger.info(f"Loading metadata for model '{model_id}' from {metadata_path}")
    try:
        with open(metadata_path) as f:
            metadata_as_dict = json.load(f)
        metadata = ClassifierMetadata(**metadata_as_dict)
        return metadata
    except json.JSONDecodeError as e:
        logger.exception(
            f"Failed to decode JSON metadata for '{model_id}' from {metadata_path}: {e}"
        )
        raise ValueError(f"Invalid JSON metadata for model '{model_id}'") from e
    except TypeError as e:
        # This catches errors if JSON keys don't match dataclass fields
        # (e.g., missing required fields or type mismatches).
        logger.exception(
            f"Mismatch between JSON structure and ClassifierMetadata "
            f"for '{model_id}': {e}"
        )
        raise ValueError(
            f"Metadata structure mismatch for model '{model_id}'. "
            f"Check required fields."
        ) from e
    except Exception as e:
        logger.exception(
            f"Failed to load metadata for '{model_id}' from {metadata_path}: {e}"
        )
        raise  # Re-raise


# --- Private Helper Functions ---


# Renamed for clarity and responsibility
def _get_models_dir_and_ensure_exists() -> Path:
    """Gets the model directory path from config and ensures it exists."""
    try:
        models_dir = get_models_root_path()  # Use the config utility
    except (FileNotFoundError, ValueError) as e:
        logger.exception(f"Cannot determine models directory: {e}")
        raise RuntimeError("Model registry path configuration error.") from e

    # Ensure the directory exists
    if not models_dir.exists():
        logger.info(f"Models directory does not exist, creating it at {models_dir}")
        try:
            models_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.exception(f"Could not create models directory {models_dir}: {e}")
            raise RuntimeError(f"Failed to create models directory {models_dir}") from e
    elif not models_dir.is_dir():
        logger.error(
            f"Configured models path exists but is not a directory: {models_dir}"
        )
        raise NotADirectoryError(
            f"Configured models path is not a directory: {models_dir}"
        )

    return models_dir


def _get_model_path(model_id: str) -> Path:
    """Constructs the full path for a model artifact file using configured base path."""
    models_dir = get_models_root_path()  # Get base path from config
    return models_dir / f"{model_id}.pkl"


def _get_model_metadata_path(model_id: str) -> Path:
    """Constructs the full path for a model metadata file using configured base path."""
    models_dir = get_models_root_path()  # Get base path from config
    return models_dir / f"{model_id}.json"
