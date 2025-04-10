from pathlib import Path

import pytest

# Import classes/functions needed for mocking/assertions
from dsba.model_registry import ClassifierMetadata
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

# Import the FastAPI app instance correctly
from src.api.api import app

# --- Constants for Test Data (assuming these are defined/correct) ---
TEST_DATA_DIR = Path("tests/data")
LOCAL_SAMPLE_CSV = TEST_DATA_DIR / "sample_training_data.csv"
# Replace with your actual raw GitHub URL once the file is pushed
GITHUB_USER = "dorianb04"  # Replace if different
REPO_NAME = "dsba-platform"  # Replace if different
BRANCH = "main"  # Or the branch where the file exists
URL_SAMPLE_CSV = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/{BRANCH}/tests/data/sample_training_data.csv"


# --- Fixtures ---


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Create a TestClient instance for the FastAPI app."""
    return TestClient(app)


# --- Test Functions ---


# Testing GET /
def test_read_root(client: TestClient):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the DSBA MLOps Platform API!"}


# Testing GET /models/
def test_list_models_success(client: TestClient, mocker: MockerFixture):
    """Test GET /models/ success."""
    expected_ids = ["model_a", "model_b"]
    mock_list = mocker.patch("src.api.api.list_models_ids", return_value=expected_ids)

    response = client.get("/models/")

    assert response.status_code == 200
    assert response.json() == expected_ids
    mock_list.assert_called_once()


def test_list_models_error(client: TestClient, mocker: MockerFixture):
    """Test GET /models/ handles unexpected errors (500)."""
    error_message = "DB connection failed"
    mock_list = mocker.patch(
        "src.api.api.list_models_ids", side_effect=Exception(error_message)
    )

    response = client.get("/models/")

    assert response.status_code == 500  # Explicit try/except in corrected API
    assert error_message in response.json()["detail"]
    mock_list.assert_called_once()


# Testing GET /models/{model_id}/metadata
def test_get_metadata_success(client: TestClient, mocker: MockerFixture):
    """Test GET /models/{model_id}/metadata success."""
    model_id = "model_a"
    # Use a dict representing the data the Pydantic model expects
    mock_meta_dict = {
        "id": model_id,
        "target_column": "target",
        "algorithm": "TestAlgo",
        "created_at": "2025-04-10T10:00:00Z",
        "hyperparameters": {"a": 1},
        "description": "Test model",
        "performance_metrics": {"acc": 0.9},
    }
    # Mock load_model_metadata to return a dataclass instance
    mock_load_meta = mocker.patch(
        "src.api.api.load_model_metadata",
        return_value=ClassifierMetadata(**mock_meta_dict),
    )

    response = client.get(f"/models/{model_id}/metadata")

    assert response.status_code == 200
    # FastAPI serializes the Pydantic response model based on the returned dataclass
    assert response.json() == mock_meta_dict
    mock_load_meta.assert_called_once_with(model_id)


def test_get_metadata_not_found(client: TestClient, mocker: MockerFixture):
    """Test GET /models/{model_id}/metadata returns 404."""
    model_id = "not_a_model"
    mock_load_meta = mocker.patch(
        "src.api.api.load_model_metadata", side_effect=FileNotFoundError("Mock 404")
    )

    response = client.get(f"/models/{model_id}/metadata")

    assert response.status_code == 404
    assert f"Model metadata not found for ID: {model_id}" in response.json()["detail"]
    mock_load_meta.assert_called_once_with(model_id)


def test_get_metadata_value_error(client: TestClient, mocker: MockerFixture):
    """Test GET /models/{model_id}/metadata returns 400 for bad metadata."""
    model_id = "bad_meta_model"
    error_message = "Invalid JSON structure"
    mock_load_meta = mocker.patch(
        "src.api.api.load_model_metadata", side_effect=ValueError(error_message)
    )

    response = client.get(f"/models/{model_id}/metadata")

    assert response.status_code == 400  # Explicitly handled in corrected API
    assert error_message in response.json()["detail"]
    mock_load_meta.assert_called_once_with(model_id)


# Testing POST /train/
def test_train_success(client: TestClient, mocker: MockerFixture):
    """Test POST /train/ success path with JSON body."""
    model_id = "new_trained_model"
    target = "target"
    # Use the actual local path for the test
    data_src = str(LOCAL_SAMPLE_CSV.resolve())  # Ensure absolute path if needed

    assert LOCAL_SAMPLE_CSV.is_file(), "Sample data file must exist for test setup"

    # Prepare request body matching TrainRequest model
    request_data = {
        "model_id": model_id,
        "data_source": data_src,
        "target_column": target,
        "test_size": 0.25,
    }

    # Mock return values for pipeline and save
    mock_model = mocker.MagicMock()
    # Ensure mock metadata matches response model structure
    mock_meta_return = ClassifierMetadata(
        id=model_id,
        target_column=target,
        algorithm="MockAlgo",
        created_at="2025-04-10T11:00:00Z",
        hyperparameters={"p": 1},
        description=f"Model trained on data from {data_src}",
        performance_metrics={"f1_score": 0.91},
    )
    mock_pipeline = mocker.patch(
        "src.api.api.run_training_pipeline", return_value=(mock_model, mock_meta_return)
    )
    mock_save = mocker.patch(
        "src.api.api.save_model", return_value=None
    )  # save_model returns None

    # Send POST request with JSON body
    response = client.post("/train/", json=request_data)

    assert response.status_code == 201  # Check for Created status
    response_json = response.json()

    # Check if response matches the structure
    # and data of ModelMetadataResponse/ClassifierMetadata
    assert response_json["id"] == model_id
    assert response_json["target_column"] == target
    assert response_json["algorithm"] == "MockAlgo"
    assert response_json["performance_metrics"]["f1_score"] == 0.91

    # Ensure underlying functions were called correctly
    mock_pipeline.assert_called_once_with(
        data_source=data_src, target_column=target, model_id=model_id, test_size=0.25
    )
    mock_save.assert_called_once_with(mock_model, mock_meta_return)


# Parameterize training failure tests
@pytest.mark.parametrize(
    "error_raised, expected_status, expected_detail_part",
    [
        (FileNotFoundError("Mocked data not found"), 400, "data file not found"),
        (
            ConnectionError("Mocked URL unreachable"),
            400,
            "Could not access training data URL",
        ),
        (ValueError("Mocked bad data"), 400, "Error during training data processing"),
        (
            RuntimeError("Mocked training execution failed"),
            500,
            "training execution failed",
        ),
        (
            Exception("Some other unexpected error"),
            500,
            "Internal server error during training",
        ),
    ],
)
def test_train_pipeline_fails(
    client: TestClient,
    mocker: MockerFixture,
    error_raised,
    expected_status,
    expected_detail_part,
):
    """Test POST /train/ handles various errors from the pipeline."""
    mock_pipeline = mocker.patch(
        "src.api.api.run_training_pipeline", side_effect=error_raised
    )
    mock_save = mocker.patch("src.api.api.save_model")

    # Minimal valid request body
    request_data = {"model_id": "fail", "data_source": "dummy", "target_column": "t"}

    response = client.post("/train/", json=request_data)

    assert response.status_code == expected_status
    assert expected_detail_part in response.json()["detail"]
    mock_pipeline.assert_called_once()  # Pipeline was called
    mock_save.assert_not_called()  # Save should not be called if pipeline fails


def test_train_invalid_request_body(client: TestClient):
    """Test POST /train/ returns 422 for invalid request body (Pydantic validation)."""
    invalid_request_data = {
        # Missing model_id, data_source, target_column
        "test_size": 1.5  # Invalid value
    }
    response = client.post("/train/", json=invalid_request_data)
    assert response.status_code == 422  # Unprocessable Entity


# Testing POST /predict/
def test_predict_success(client: TestClient, mocker: MockerFixture):
    """Test POST /predict/ success path with JSON body."""
    model_id = "predictor_model"
    features_dict = {"featureA": 10.5, "featureB": "value2"}
    expected_prediction = 0

    # Mock underlying functions
    mock_model_obj = mocker.MagicMock()
    mock_load_model = mocker.patch(
        "src.api.api.load_model", return_value=mock_model_obj
    )
    # Mock metadata only needed if classify_record uses it
    mock_meta = ClassifierMetadata(id=model_id, target_column="target")
    mock_load_meta = mocker.patch(
        "src.api.api.load_model_metadata", return_value=mock_meta
    )
    mock_classify = mocker.patch(
        "src.api.api.classify_record", return_value=expected_prediction
    )

    # Prepare request body matching PredictRequest
    request_data = {"model_id": model_id, "features": features_dict}

    response = client.post("/predict/", json=request_data)

    assert response.status_code == 200
    assert response.json() == {"model_id": model_id, "prediction": expected_prediction}
    # Verify calls
    mock_load_model.assert_called_once_with(model_id)
    mock_load_meta.assert_called_once_with(model_id)
    mock_classify.assert_called_once_with(mock_model_obj, features_dict, "target")


def test_predict_model_not_found(client: TestClient, mocker: MockerFixture):
    """Test POST /predict/ returns 404 if model not found."""
    model_id = "ghost_predictor"
    request_data = {"model_id": model_id, "features": {"f": 1}}
    mock_load_model = mocker.patch(
        "src.api.api.load_model", side_effect=FileNotFoundError("Not here")
    )
    mock_load_meta = mocker.patch(
        "src.api.api.load_model_metadata"
    )  # Mock just in case
    mock_classify = mocker.patch("src.api.api.classify_record")

    response = client.post("/predict/", json=request_data)

    assert response.status_code == 404  # Explicitly handled in corrected API
    assert (
        f"Model (or its metadata) not found for ID: {model_id}"
        in response.json()["detail"]
    )
    mock_load_model.assert_called_once_with(model_id)
    mock_load_meta.assert_not_called()  # Should fail before loading metadata
    mock_classify.assert_not_called()


def test_predict_value_error(client: TestClient, mocker: MockerFixture):
    model_id = "value_err_model"
    request_data = {"model_id": model_id, "features": {"f": "bad_type"}}

    mocker.patch("src.api.api.load_model", return_value=mocker.MagicMock())
    mocker.patch("src.api.api.load_model_metadata", return_value=mocker.MagicMock())
    error_message = "Feature 'f' has invalid type"
    mock_classify = mocker.patch(
        "src.api.api.classify_record", side_effect=ValueError(error_message)
    )

    response = client.post("/predict/", json=request_data)

    assert response.status_code == 400
    assert error_message in response.json()["detail"]
    mock_classify.assert_called_once()


def test_predict_invalid_request_body(client: TestClient):
    """Test POST /predict/ returns 422 for invalid JSON body."""
    # Missing 'features' field
    invalid_request_data = {"model_id": "some_model"}
    response = client.post("/predict/", json=invalid_request_data)
    assert response.status_code == 422  # Pydantic validation error

    # Invalid data type for features (should be dict)
    invalid_request_data_2 = {
        "model_id": "some_model",
        "features": ["list", "not", "dict"],
    }
    response_2 = client.post("/predict/", json=invalid_request_data_2)
    assert response_2.status_code == 422
