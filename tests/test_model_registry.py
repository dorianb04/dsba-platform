import json
from pathlib import Path

# Import the function we need to mock
import pytest
from dsba.model_registry import (
    ClassifierMetadata,
    list_models_ids,
    load_model,
    load_model_metadata,
    save_model,
)
from pytest_mock import MockerFixture
from sklearn.linear_model import LogisticRegression

MOCK_TARGET = "dsba.model_registry.get_models_root_path"

# Test Suite for Model Registry Operations


@pytest.fixture(scope="function")
def mock_registry_path(tmp_path: Path, mocker: MockerFixture):
    """
    Mocks the get_models_root_path function used by model_registry
    to return a temporary directory for isolated testing.
    """
    models_dir = tmp_path / "test_registry"
    models_dir.mkdir()

    mocker.patch(MOCK_TARGET, return_value=models_dir)

    yield models_dir


# Fixture for a sample model and metadata
@pytest.fixture
def sample_model_and_metadata():
    model = LogisticRegression()
    metadata = ClassifierMetadata(
        id="test_model_01",
        target_column="target",
        algorithm="LogisticRegression",
        description="A simple test model",
        hyperparameters={"C": 1.0, "solver": "lbfgs"},
        performance_metrics={"accuracy": 0.9},
    )
    return model, metadata


def test_save_model_creates_files(mock_registry_path: Path, sample_model_and_metadata):
    """Test that save_model creates .pkl and .json files in the mocked path."""
    model, metadata = sample_model_and_metadata
    registry_path = mock_registry_path

    # save_model internally calls the mocked get_models_root_path()
    save_model(model, metadata)

    model_file = registry_path / f"{metadata.id}.pkl"
    metadata_file = registry_path / f"{metadata.id}.json"

    # Assertions should now work against the temp directory
    assert model_file.exists()
    assert model_file.is_file()
    assert metadata_file.exists()
    assert metadata_file.is_file()

    # Verify metadata content
    with open(metadata_file) as f:
        saved_meta = json.load(f)
        assert saved_meta["id"] == metadata.id
        assert saved_meta["hyperparameters"] == metadata.hyperparameters


def test_load_model_success(mock_registry_path: Path, sample_model_and_metadata):
    """Test loading a previously saved model from the mocked path."""
    model, metadata = sample_model_and_metadata
    # Ensure save_model uses the mocked path by calling it within the test
    save_model(model, metadata)

    # load_model should now use the mocked path internally
    loaded_model = load_model(metadata.id)
    assert isinstance(loaded_model, LogisticRegression)
    assert loaded_model.get_params()["C"] == 1.0


def test_load_model_not_found(mock_registry_path: Path):
    """Test loading a non-existent model raises FileNotFoundError from mocked path."""
    # Mock ensures we look in the temp dir, where the model doesn't exist
    with pytest.raises(FileNotFoundError):
        load_model("non_existent_model")


def test_load_metadata_success(mock_registry_path: Path, sample_model_and_metadata):
    """Test loading previously saved metadata from the mocked path."""
    model, metadata = sample_model_and_metadata
    save_model(model, metadata)  # Save first using mocked path

    # Load should use mocked path
    loaded_metadata = load_model_metadata(metadata.id)
    assert isinstance(loaded_metadata, ClassifierMetadata)
    assert loaded_metadata.__dict__ == metadata.__dict__


def test_load_metadata_not_found(mock_registry_path: Path):
    """Test loading non-existent metadata raises FileNotFoundError from mocked path."""
    with pytest.raises(FileNotFoundError):
        load_model_metadata("non_existent_model")


def test_load_metadata_missing_optional_fields(mock_registry_path: Path):
    """Test loading metadata with missing optional fields from mocked path."""
    registry_path = mock_registry_path
    model_id = "minimal_model"
    minimal_meta_dict = {
        "id": model_id,
        "target_column": "minimal_target",
    }
    metadata_file = registry_path / f"{model_id}.json"
    with open(metadata_file, "w") as f:
        json.dump(minimal_meta_dict, f)

    # Create a dummy .pkl file in the mocked path
    (registry_path / f"{model_id}.pkl").touch()

    # Load should now correctly find and parse the file in the temp dir
    loaded_metadata = load_model_metadata(model_id)
    assert loaded_metadata.id == model_id
    assert loaded_metadata.target_column == "minimal_target"
    assert loaded_metadata.algorithm == "Unknown"
    assert loaded_metadata.created_at is not None


def test_list_models_ids(mock_registry_path: Path, sample_model_and_metadata):
    """Test listing model IDs from the mocked path."""
    registry_path = mock_registry_path  # Get mocked path for clarity

    # list_models_ids should use the mocked path internally
    assert list_models_ids() == []

    # Save one model (uses mocked path)
    model, metadata = sample_model_and_metadata
    save_model(model, metadata)
    assert list_models_ids() == [metadata.id]

    # Save another model
    model2 = LogisticRegression(C=0.5)
    metadata2 = ClassifierMetadata(id="test_model_02", target_column="target2")
    save_model(model2, metadata2)  # Uses mocked path

    assert sorted(list_models_ids()) == sorted([metadata.id, metadata2.id])

    # Check that non-model files in the temp dir are ignored
    (registry_path / "not_a_model.txt").touch()
    assert sorted(list_models_ids()) == sorted([metadata.id, metadata2.id])
