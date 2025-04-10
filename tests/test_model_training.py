from pathlib import Path

# Import the specific module where requests is used for correct patching
import numpy as np
import pytest
import requests  # Import requests for exception types

# Mock imports
from dsba.model_registry import ClassifierMetadata

# Import the main function and supporting classes/types
from dsba.model_training import _create_metadata, run_training_pipeline
from pytest_mock import MockerFixture

# --- Constants for Test Data ---
TEST_DATA_DIR = Path("tests/data")
LOCAL_SAMPLE_CSV = TEST_DATA_DIR / "sample_training_data.csv"
GITHUB_USER = "dorianb04"  # Replace if different
REPO_NAME = "dsba-platform"  # Replace if different
BRANCH = "main"  # Or the branch where the file exists
URL_SAMPLE_CSV = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/{BRANCH}/tests/data/sample_training_data.csv"


# --- Simplified Tests Focusing on Pipeline ---


def test_pipeline_local_file_success(mocker: MockerFixture):
    """
    Tests the pipeline success path using the local sample CSV.
    Mocks only the _train_model and _evaluate_model helper functions.
    """
    model_id = "pipeline_local_success"
    target_col = "target"
    data_source = str(LOCAL_SAMPLE_CSV)

    assert LOCAL_SAMPLE_CSV.is_file(), f"Test data file not found at {LOCAL_SAMPLE_CSV}"

    # --- Arrange Mocks ---
    # 1. Mock _train_model helper
    mock_model_obj = mocker.MagicMock()
    mock_model_obj.get_params.return_value = {"mock_param": 1}
    mock_model_obj.__class__.__name__ = "MockedTrainer"
    mock_model_obj._estimator_type = "classifier"
    # Mock the predict method as it's called indirectly by _evaluate_model's dependency
    mock_model_obj.predict.return_value = np.array(
        [0, 1, 0]
    )  # Dummy predictions for test_size=0.3
    mock_train = mocker.patch(
        "dsba.model_training._train_model", return_value=mock_model_obj
    )

    # 2. Mock _evaluate_model helper
    mock_metrics = {
        "accuracy": 0.95,
        "f1_score": 0.90,
        "precision": 0.85,
        "recall": 0.92,
    }
    mock_eval = mocker.patch(
        "dsba.model_training._evaluate_model", return_value=mock_metrics
    )

    # 3. Mock _create_metadata using wraps= with the correctly imported function
    #    This allows us to check its input while still running its actual logic.
    mock_create_meta = mocker.patch(
        "dsba.model_training._create_metadata", wraps=_create_metadata
    )

    # --- Act ---
    model, metadata = run_training_pipeline(
        data_source=data_source,
        target_column=target_col,
        model_id=model_id,
        test_size=0.3,  # Match prediction mock size if needed
    )

    # --- Assert ---
    mock_train.assert_called_once()
    mock_eval.assert_called_once()
    # Check the metrics dict passed to the (wrapped) _create_metadata mock
    assert mock_create_meta.call_args[0][4] == mock_metrics

    assert model == mock_model_obj
    assert isinstance(metadata, ClassifierMetadata)
    assert metadata.id == model_id
    assert (
        metadata.performance_metrics == mock_metrics
    )  # Check the actual metadata object


def test_pipeline_local_file_not_found():
    """
    Tests pipeline failure when the local data file doesn't exist.
    """
    model_id = "pipeline_file_not_found"
    target_col = "target"
    data_source = "path/to/non/existent/file.csv"

    with pytest.raises(FileNotFoundError):
        run_training_pipeline(
            data_source=data_source, target_column=target_col, model_id=model_id
        )


def test_pipeline_missing_target_column():
    """
    Tests pipeline failure when target column is missing.
    """
    model_id = "pipeline_missing_target"
    target_col = "non_existent_target"
    data_source = str(LOCAL_SAMPLE_CSV)

    assert LOCAL_SAMPLE_CSV.is_file(), f"Test data file not found at {LOCAL_SAMPLE_CSV}"

    with pytest.raises(ValueError, match=f"Target column '{target_col}' not found"):
        run_training_pipeline(
            data_source=data_source, target_column=target_col, model_id=model_id
        )


def test_pipeline_url_success(mocker: MockerFixture):
    """
    Tests the pipeline success path using a URL.
    Mocks requests.get and the _train/_evaluate helpers.
    """
    model_id = "pipeline_url_success"
    target_col = "target"
    data_source = URL_SAMPLE_CSV

    # 1. Mock requests.get where it's used (dsba.data_ingestion.files)
    mock_response = mocker.MagicMock()
    mock_response.status_code = 200
    mock_response.text = LOCAL_SAMPLE_CSV.read_text()
    mock_req_get = mocker.patch(
        "dsba.data_ingestion.files.requests.get", return_value=mock_response
    )  # Corrected Target

    # 2. Mock the ML helper functions
    mock_model_obj = mocker.MagicMock()
    mock_model_obj.get_params.return_value = {"mock_param": 1}
    mock_model_obj.__class__.__name__ = "MockedTrainerURL"
    mock_model_obj._estimator_type = "classifier"
    mock_model_obj.predict.return_value = np.array([1, 1, 0])
    mock_train = mocker.patch(
        "dsba.model_training._train_model", return_value=mock_model_obj
    )
    mock_metrics = {"accuracy": 0.88, "f1_score": 0.80}
    mock_eval = mocker.patch(
        "dsba.model_training._evaluate_model", return_value=mock_metrics
    )
    mock_create_meta = mocker.patch(
        "dsba.model_training._create_metadata", wraps=_create_metadata
    )

    # --- Act ---
    model, metadata = run_training_pipeline(
        data_source=data_source,
        target_column=target_col,
        model_id=model_id,
        test_size=0.3,
    )

    # --- Assert ---
    mock_req_get.assert_called_once_with(data_source)  # Verify URL load attempt
    mock_train.assert_called_once()
    mock_eval.assert_called_once()
    assert mock_create_meta.call_args[0][4] == mock_metrics
    assert model == mock_model_obj
    assert isinstance(metadata, ClassifierMetadata)
    assert metadata.performance_metrics == mock_metrics


def test_pipeline_url_fails(mocker: MockerFixture):
    """
    Tests pipeline failure when URL is invalid.
    Mocks requests.get to raise an error.
    """
    model_id = "pipeline_url_fail"
    target_col = "target"
    data_source = "http://invalid-url-for-testing"

    # Mock requests.get where it's used in data_ingestion.files
    mock_req_get = mocker.patch(
        "dsba.data_ingestion.files.requests.get",
        side_effect=requests.exceptions.ConnectionError("Mock connection error"),
    )

    expected_error_msg = f"Could not connect/read URL {data_source}"

    with pytest.raises(ConnectionError, match=expected_error_msg):
        run_training_pipeline(
            data_source=data_source, target_column=target_col, model_id=model_id
        )
    mock_req_get.assert_called_once_with(data_source)
