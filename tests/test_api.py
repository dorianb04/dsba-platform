from pathlib import Path

import pytest

# Import classes/functions needed for mocking/assertions
from dsba.model_registry import ClassifierMetadata
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

# Import the FastAPI app instance correctly
from src.api.api import app

# --- Constants for Test Data ---
TEST_DATA_DIR = Path("tests/data")
LOCAL_SAMPLE_CSV = TEST_DATA_DIR / "sample_training_data.csv"
GITHUB_USER = "dorianb04"
REPO_NAME = "dsba-platform"
BRANCH = "main"
URL_SAMPLE_CSV = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/{BRANCH}/tests/data/sample_training_data.csv"


# --- Fixtures ---


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Create a TestClient instance for the FastAPI app."""
    return TestClient(app)


# --- Test Functions ---


# Testing GET /
def test_read_root(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the DSBA MLOps Platform API!"}


# Testing GET /models/
def test_list_models_success(client: TestClient, mocker: MockerFixture):
    expected_ids = ["model_a", "model_b"]
    mock_list = mocker.patch("src.api.api.list_models_ids", return_value=expected_ids)
    response = client.get("/models/")
    assert response.status_code == 200
    assert response.json() == expected_ids
    mock_list.assert_called_once()


def test_list_models_error(client: TestClient, mocker: MockerFixture):
    error_message = "DB connection failed"
    mock_list = mocker.patch(
        "src.api.api.list_models_ids", side_effect=Exception(error_message)
    )
    response = client.get("/models/")
    assert response.status_code == 500
    assert error_message in response.json()["detail"]
    mock_list.assert_called_once()


# Testing GET /models/{model_id}/metadata
def test_get_metadata_success(client: TestClient, mocker: MockerFixture):
    model_id = "model_a"
    mock_meta_dict = {
        "id": model_id,
        "target_column": "target",
        "algorithm": "TestAlgo",
        "created_at": "ts",
        "hyperparameters": {"a": 1},
        "description": "d",
        "performance_metrics": {"acc": 0.9},
    }
    mock_load_meta = mocker.patch(
        "src.api.api.load_model_metadata",
        return_value=ClassifierMetadata(**mock_meta_dict),
    )
    response = client.get(f"/models/{model_id}/metadata")
    assert response.status_code == 200

    assert response.json()["id"] == model_id
    assert response.json()["algorithm"] == "TestAlgo"
    mock_load_meta.assert_called_once_with(model_id)


def test_get_metadata_not_found(client: TestClient, mocker: MockerFixture):
    model_id = "not_a_model"
    mock_load_meta = mocker.patch(
        "src.api.api.load_model_metadata", side_effect=FileNotFoundError("Mock 404")
    )
    response = client.get(f"/models/{model_id}/metadata")
    assert response.status_code == 404
    assert f"Model metadata not found for ID: {model_id}" in response.json()["detail"]
    mock_load_meta.assert_called_once_with(model_id)


def test_get_metadata_value_error(client: TestClient, mocker: MockerFixture):
    model_id = "bad_meta_model"
    error_message = "Invalid JSON structure"
    mock_load_meta = mocker.patch(
        "src.api.api.load_model_metadata", side_effect=ValueError(error_message)
    )
    response = client.get(f"/models/{model_id}/metadata")
    assert response.status_code == 400
    assert error_message in response.json()["detail"]
    mock_load_meta.assert_called_once_with(model_id)


# Testing POST /train/
def test_train_success(client: TestClient, mocker: MockerFixture):
    model_id = "new_trained_model"
    target = "target"
    data_src = str(LOCAL_SAMPLE_CSV.resolve())
    assert LOCAL_SAMPLE_CSV.is_file(), "Sample data file must exist"
    request_data = {
        "model_id": model_id,
        "data_source": data_src,
        "target_column": target,
        "test_size": 0.25,
    }
    mock_model = mocker.MagicMock()
    mock_meta_return = ClassifierMetadata(
        id=model_id,
        target_column=target,
        algorithm="MockAlgo",
        created_at="ts",
        hyperparameters={},
        description="",
        performance_metrics={"f1_score": 0.91},
    )
    mock_pipeline = mocker.patch(
        "src.api.api.run_training_pipeline", return_value=(mock_model, mock_meta_return)
    )
    mock_save = mocker.patch("src.api.api.save_model", return_value=None)

    response = client.post("/train/", json=request_data)

    assert response.status_code == 201
    response_json = response.json()
    assert response_json["id"] == model_id
    assert response_json["performance_metrics"]["f1_score"] == 0.91
    mock_pipeline.assert_called_once_with(
        data_source=data_src, target_column=target, model_id=model_id, test_size=0.25
    )
    mock_save.assert_called_once_with(mock_model, mock_meta_return)


# Parameterized training failure tests
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
    mock_pipeline = mocker.patch(
        "src.api.api.run_training_pipeline", side_effect=error_raised
    )
    mock_save = mocker.patch("src.api.api.save_model")
    request_data = {"model_id": "fail", "data_source": "dummy", "target_column": "t"}
    response = client.post("/train/", json=request_data)
    assert response.status_code == expected_status
    assert expected_detail_part in response.json()["detail"]
    mock_pipeline.assert_called_once()
    mock_save.assert_not_called()


def test_train_invalid_request_body(client: TestClient):
    invalid_request_data = {"test_size": 1.5}
    response = client.post("/train/", json=invalid_request_data)
    assert response.status_code == 422


# --- Testing POST /predict/ ---


def test_predict_success_multiple(client: TestClient, mocker: MockerFixture):
    """Test POST /predict/ success path with multiple feature sets."""
    model_id = "predictor_model"
    feature_sets = [
        {"featureA": 10.5, "featureB": "value1"},
        {"featureA": 20.7, "featureB": "value2"},
    ]
    expected_predictions = [0, 1]

    # Mock underlying functions
    mock_model_obj = mocker.MagicMock()
    mock_load_model = mocker.patch(
        "src.api.api.load_model", return_value=mock_model_obj
    )
    mock_meta = ClassifierMetadata(id=model_id, target_column="target")
    mock_load_meta = mocker.patch(
        "src.api.api.load_model_metadata", return_value=mock_meta
    )

    # Mock classify_record to return values sequentially
    mock_classify = mocker.patch("src.api.api.classify_record")
    mock_classify.side_effect = expected_predictions

    request_data = {"model_id": model_id, "features": feature_sets}

    response = client.post("/predict/", json=request_data)

    assert response.status_code == 200
    response_json = response.json()
    assert isinstance(response_json, list)
    assert len(response_json) == len(expected_predictions)

    # Check each prediction in the response list
    for i, pred_response in enumerate(response_json):
        assert pred_response["model_id"] == model_id
        assert pred_response["prediction"] == expected_predictions[i]

    # Verify calls
    mock_load_model.assert_called_once_with(model_id)
    mock_load_meta.assert_called_once_with(model_id)
    assert mock_classify.call_count == len(feature_sets)
    mock_classify.assert_any_call(mock_model_obj, feature_sets[0], "target")
    mock_classify.assert_any_call(mock_model_obj, feature_sets[1], "target")


def test_predict_model_not_found(client: TestClient, mocker: MockerFixture):
    """Test POST /predict/ returns 404 if model not found."""
    model_id = "ghost_predictor"
    request_data = {"model_id": model_id, "features": [{"f": 1}]}
    mock_load_model = mocker.patch(
        "src.api.api.load_model", side_effect=FileNotFoundError("Not here")
    )
    mock_load_meta = mocker.patch("src.api.api.load_model_metadata")
    mock_classify = mocker.patch("src.api.api.classify_record")

    response = client.post("/predict/", json=request_data)

    assert response.status_code == 404
    assert (
        f"Model (or its metadata) not found for ID: {model_id}"
        in response.json()["detail"]
    )
    mock_load_model.assert_called_once_with(model_id)
    mock_load_meta.assert_not_called()
    mock_classify.assert_not_called()


def test_predict_value_error(client: TestClient, mocker: MockerFixture):
    """Test POST /predict/ returns 400 if classification raises ValueError."""
    model_id = "value_err_model"
    request_data = {"model_id": model_id, "features": [{"f": "bad_type"}]}
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
    """Test POST /predict/ returns 422 for invalid JSON body structure."""
    # Missing 'features' field
    invalid_request_data = {"model_id": "some_model"}
    response = client.post("/predict/", json=invalid_request_data)
    assert response.status_code == 422

    # Invalid data type for features (should be list)
    invalid_request_data_2 = {"model_id": "some_model", "features": "not-a-list"}
    response_2 = client.post("/predict/", json=invalid_request_data_2)
    assert response_2.status_code == 422

    # Invalid type inside features list (should be dict)
    invalid_request_data_3 = {"model_id": "some_model", "features": ["not-a-dict"]}
    response_3 = client.post("/predict/", json=invalid_request_data_3)
    assert response_3.status_code == 422
