from pathlib import Path

import pandas as pd
import pytest

# Only import specific exceptions if checking for them by type explicitly
# Import classes/functions from dsba library that are needed for MOCKING
from dsba.model_registry import (
    ClassifierMetadata,
)
from pytest_mock import MockerFixture

from src.cli.dsba_cli import (
    create_parser,
    list_models_cli,
    predict_batch_cli,
    show_metadata_cli,
    train_model_cli,
)

# --- Fixtures ---


@pytest.fixture(scope="function")
def mock_cli_config(tmp_path: Path, mocker: MockerFixture):
    """Mocks config functions used by library calls needed by CLI functions."""
    models_dir = tmp_path / "cli_test_registry"
    models_dir.mkdir()
    mocker.patch("dsba.config.get_models_root_path", return_value=models_dir)
    mocker.patch("dsba.config.load_config", return_value={})
    yield models_dir


@pytest.fixture
def sample_cli_csv_file(tmp_path: Path) -> Path:
    """Creates a sample CSV file in a temp dir for CLI tests."""
    data = {"featureA": [1, 2, 3], "target": [0, 1, 0]}
    df = pd.DataFrame(data)
    csv_path = tmp_path / "cli_sample_data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# --- Basic Argparse Test ---


def test_cli_parser_defines_commands():
    """Check if main commands are defined."""
    parser = create_parser()
    parser.parse_args(["list"])
    parser.parse_args(["metadata", "some_id"])
    parser.parse_args(
        ["train", "--model-id", "id", "--data-source", "src", "--target-column", "t"]
    )
    parser.parse_args(
        ["predict", "--model-id", "id", "--input", "in.csv", "--output", "out.csv"]
    )


def test_cli_train_requires_args():
    """Check train command requires arguments using parse_args."""
    parser = create_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["train", "--model-id", "m"])


# --- Test Command Functions Directly ---


def test_list_models_cli_success(mock_cli_config: Path, mocker: MockerFixture, capsys):
    """Test list_models_cli prints expected output."""
    expected_models = ["model1", "model2"]
    # Mock the library function CALLED BY list_models_cli
    mock_list_ids = mocker.patch(
        "src.cli.dsba_cli.list_models_ids", return_value=expected_models
    )

    list_models_cli()
    captured = capsys.readouterr()

    mock_list_ids.assert_called_once()
    assert "Available models:" in captured.out
    assert "- model1" in captured.out
    assert "- model2" in captured.out


def test_list_models_cli_empty(mock_cli_config: Path, mocker: MockerFixture, capsys):
    """Test list_models_cli handles empty list."""
    mock_list_ids = mocker.patch("src.cli.dsba_cli.list_models_ids", return_value=[])
    list_models_cli()
    captured = capsys.readouterr()
    mock_list_ids.assert_called_once()
    assert "No models found" in captured.out


def test_show_metadata_cli_success(
    mock_cli_config: Path, mocker: MockerFixture, capsys
):
    """Test show_metadata_cli prints formatted metadata."""
    model_id = "meta_model"
    mock_meta_obj = ClassifierMetadata(
        id=model_id,
        target_column="t",
        algorithm="Algo",
        created_at="ts",
        hyperparameters={"p": 1},
        description="d",
        performance_metrics={"f1": 0.5},
    )
    # Mock the library function CALLED BY show_metadata_cli
    mock_load = mocker.patch(
        "src.cli.dsba_cli.load_model_metadata", return_value=mock_meta_obj
    )
    # Mock json.dumps used inside show_metadata_cli for predictable output checks
    mocker.patch(
        "src.cli.dsba_cli.json.dumps", return_value='{"algorithm": "Algo", "f1": 0.5}'
    )

    show_metadata_cli(model_id)
    captured = capsys.readouterr()

    mock_load.assert_called_once_with(model_id)
    assert f"--- Metadata for Model: {model_id} ---" in captured.out
    assert '{"algorithm": "Algo", "f1": 0.5}' in captured.out


def test_show_metadata_cli_not_found(mock_cli_config: Path, mocker: MockerFixture):
    """Test show_metadata_cli handles FileNotFoundError from load_model_metadata."""
    model_id = "ghost"
    error_msg = f"Metadata file not found for model ID '{model_id}'"
    # Mock the library function CALLED BY show_metadata_cli
    mock_load = mocker.patch(
        "src.cli.dsba_cli.load_model_metadata", side_effect=FileNotFoundError(error_msg)
    )

    # Check if the function raises the specific error it's supposed to
    with pytest.raises(FileNotFoundError, match=error_msg):
        show_metadata_cli(model_id)
    mock_load.assert_called_once_with(model_id)


def test_train_model_cli_success(
    mock_cli_config: Path, sample_cli_csv_file: Path, mocker: MockerFixture, capsys
):
    """Test train_model_cli success path."""
    model_id = "train_cli_ok"
    target = "target"
    data_src = str(sample_cli_csv_file)
    test_size = 0.25
    mock_model = mocker.MagicMock()
    mock_meta = ClassifierMetadata(
        id=model_id, target_column=target, performance_metrics={"f1_score": 0.98}
    )
    # Mock the library functions CALLED BY train_model_cli
    mock_pipeline = mocker.patch(
        "src.cli.dsba_cli.run_training_pipeline", return_value=(mock_model, mock_meta)
    )
    mock_save = mocker.patch("src.cli.dsba_cli.save_model", return_value=None)
    # Mock json.dumps used inside train_model_cli
    mocker.patch("src.cli.dsba_cli.json.dumps", return_value='{"f1_score": 0.98}')

    train_model_cli(model_id, data_src, target, test_size)  # Call directly
    captured = capsys.readouterr()

    mock_pipeline.assert_called_once_with(
        data_source=data_src,
        target_column=target,
        model_id=model_id,
        test_size=test_size,
    )
    mock_save.assert_called_once_with(mock_model, mock_meta)
    assert f"Model '{model_id}' trained and saved successfully" in captured.out
    assert '{"f1_score": 0.98}' in captured.out  # Check mocked json output


def test_train_model_cli_fails(
    mock_cli_config: Path, sample_cli_csv_file: Path, mocker: MockerFixture
):
    """Test train_model_cli handles error from run_training_pipeline."""
    error_msg = "Pipeline failure"
    # Mock the library function CALLED BY train_model_cli
    mock_pipeline = mocker.patch(
        "src.cli.dsba_cli.run_training_pipeline", side_effect=RuntimeError(error_msg)
    )
    mock_save = mocker.patch("src.cli.dsba_cli.save_model")  # Mock save as well

    # Check if the function raises the specific error it's supposed to
    with pytest.raises(RuntimeError, match=f"Training process failed: {error_msg}"):
        train_model_cli("id", str(sample_cli_csv_file), "target", 0.2)

    mock_pipeline.assert_called_once()
    mock_save.assert_not_called()  # Ensure save wasn't called


def test_predict_batch_cli_success(
    mock_cli_config: Path, sample_cli_csv_file: Path, mocker: MockerFixture, capsys
):
    """Test predict_batch_cli success path."""
    model_id = "predict_cli_ok"
    input_path = sample_cli_csv_file
    output_path = mock_cli_config.parent / "predict_cli_output.csv"
    mock_model = mocker.MagicMock()
    mock_meta = ClassifierMetadata(id=model_id, target_column="target")
    mock_df = pd.DataFrame({"featureA": [1], "target": [0]})
    mock_preds = mock_df.copy()
    mock_preds["target"] = [1]

    # Mock all library functions CALLED BY predict_batch_cli
    mock_load_model = mocker.patch(
        "src.cli.dsba_cli.load_model", return_value=mock_model
    )
    mock_load_meta = mocker.patch(
        "src.cli.dsba_cli.load_model_metadata", return_value=mock_meta
    )
    mock_load_csv = mocker.patch(
        "src.cli.dsba_cli.load_csv_from_path", return_value=mock_df
    )
    mock_classify = mocker.patch(
        "src.cli.dsba_cli.classify_dataframe", return_value=mock_preds
    )
    mock_write_csv = mocker.patch(
        "src.cli.dsba_cli.write_csv_to_path", return_value=None
    )

    predict_batch_cli(model_id, input_path, output_path)  # Call directly
    captured = capsys.readouterr()

    mock_load_model.assert_called_once_with(model_id)
    mock_load_meta.assert_called_once_with(model_id)
    mock_load_csv.assert_called_once_with(input_path)
    mock_classify.assert_called_once()  # Could add arg checks if needed
    mock_write_csv.assert_called_once_with(mock_preds, output_path)
    assert f"Predictions saved to {output_path}" in captured.out
    assert f"Scored {len(mock_preds)} records" in captured.out


def test_predict_batch_cli_input_not_found(mock_cli_config: Path):
    """Test predict_batch_cli handles input FileNotFoundError."""
    input_path = Path("non/existent/input.csv")
    output_path = mock_cli_config.parent / "output.csv"
    # This error is raised inside predict_batch_cli before main handler
    with pytest.raises(FileNotFoundError):
        predict_batch_cli("any_model", input_path, output_path)
