#!python3

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

from dsba.config import get_models_root_path
from dsba.data_ingestion import load_csv_from_path, write_csv_to_path
from dsba.model_prediction import classify_dataframe
from dsba.model_registry import (
    ClassifierMetadata,
    list_models_ids,
    load_model,
    load_model_metadata,
    save_model,
)
from dsba.model_training import run_training_pipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S,",
)
logger = logging.getLogger("dsba_cli")

# --- Argument Parsing ---


def create_parser():
    """Creates the argument parser for the CLI tool."""
    parser = argparse.ArgumentParser(
        description="DSBA Platform CLI Tool - Manage and use ML models locally.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available commands"
    )

    # --- List Models Command ---
    subparsers.add_parser("list", help="List available models in the local registry.")

    # --- Get Model Metadata Command ---
    meta_parser = subparsers.add_parser(
        "metadata", help="Show metadata for a specific model."
    )
    meta_parser.add_argument("model_id", help="ID of the model.")

    # --- Train Model Command (Corrected Arguments) ---
    train_parser = subparsers.add_parser(
        "train", help="Train a new model locally and save to registry."
    )
    train_parser.add_argument(
        "--model-id", required=True, help="Unique ID to assign to the new model."
    )
    train_parser.add_argument(
        "--data-source",
        required=True,
        help="URL or local file path to the CSV training data.",
    )
    train_parser.add_argument(
        "--target-column", required=True, help="Name of the target variable column."
    )
    train_parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data for the test set (0 < size < 1).",
    )

    # --- Predict Command ---
    predict_parser = subparsers.add_parser(
        "predict", help="Make predictions on a local batch CSV file."
    )
    predict_parser.add_argument(
        "--model-id", required=True, help="ID of the model to use."
    )
    # Use type=Path for automatic conversion and validation
    predict_parser.add_argument(
        "--input", required=True, type=Path, help="Input CSV file path for prediction."
    )
    predict_parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output CSV file path to save predictions.",
    )

    return parser


# --- Command Functions (with Error Handling) ---


def list_models_cli() -> None:
    """Handles the 'list' command."""
    logger.info("Listing available models...")
    try:
        models = list_models_ids()
        if not models:
            print("No models found in the registry.")
            return
        print("Available models:")
        for model_id in sorted(models):
            print(f"- {model_id}")
    except Exception as e:
        logger.error(f"Failed to list models: {e}", exc_info=True)
        raise RuntimeError(f"Could not list models: {e}") from e


def show_metadata_cli(model_id: str) -> None:
    """Handles the 'metadata' command."""
    logger.info(f"Fetching metadata for model: {model_id}")
    try:
        metadata: ClassifierMetadata = load_model_metadata(model_id)
        print(f"--- Metadata for Model: {model_id} ---")

        print(json.dumps(dataclasses.asdict(metadata), indent=4, default=str))
        print("--- End Metadata ---")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Metadata file not found for model ID '{model_id}'"
        ) from e
    except Exception as e:
        logger.error(
            f"Failed to load/display metadata for {model_id}: {e}", exc_info=True
        )
        raise RuntimeError(f"Could not load/display metadata: {e}") from e


def train_model_cli(
    model_id: str, data_source: str, target_column: str, test_size: float
) -> None:
    """Handles the 'train' command."""
    logger.info(f"Starting training via CLI for model: {model_id}")
    print(f"Training model '{model_id}' from '{data_source}'...")
    try:
        # Call the pipeline function
        model, metadata = run_training_pipeline(
            data_source=data_source,
            target_column=target_column,
            model_id=model_id,
            test_size=test_size,
        )
        # save the model
        save_model(model, metadata)
        print(f"\nModel '{model_id}' trained and saved successfully.")
        print("\nPerformance Metrics:")
        # Pretty print just the metrics part of the metadata
        print(json.dumps(metadata.performance_metrics, indent=4, default=str))
    except Exception as e:
        # Catch errors from pipeline or save_model
        logger.error(f"Training failed for model '{model_id}': {e}", exc_info=True)
        # Re-raise wrapped error
        raise RuntimeError(f"Training process failed: {e}") from e


def predict_batch_cli(model_id: str, input_path: Path, output_path: Path) -> None:
    """Handles the 'predict' command for batch files."""
    logger.info(f"Starting batch prediction using model: {model_id}")
    print(f"Predicting using model '{model_id}'...")
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")

    try:
        # Input validation already partially done by argparse type=Path
        if not input_path.is_file():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        if output_path.exists() and not output_path.is_file():
            raise ValueError(f"Output path exists but is not a file: {output_path}")
        output_path.parent.mkdir(
            parents=True, exist_ok=True
        )  # Ensure output dir exists

        model = load_model(model_id)
        # Metadata might be needed by classify_dataframe depending on its implementation
        metadata = load_model_metadata(model_id)
        logger.info(f"Loading input data from: {input_path}")
        input_df = load_csv_from_path(input_path)

        if input_df.empty:
            logger.warning("Input dataframe for prediction is empty.")
            # Decide how to handle empty input - here we write an empty output
            predictions_df = input_df.copy()  # Keep structure but no rows
            predictions_df[metadata.target_column] = None  # Add target column header
        else:
            logger.info("Classifying dataframe...")
            predictions_df = classify_dataframe(
                model, input_df.copy(), metadata.target_column
            )

        logger.info(f"Saving predictions to: {output_path}")
        write_csv_to_path(predictions_df, output_path)
        print(f"\nPredictions saved to {output_path}.")
        print(f"Scored {len(predictions_df)} records.")

    except (
        FileNotFoundError
    ) as e:  # Catch file errors from load_model, load_metadata, load_csv
        raise e  # Re-raise for main handler
    except Exception as e:
        logger.error(f"Prediction failed for model '{model_id}': {e}", exc_info=True)
        raise RuntimeError(f"Prediction process failed: {e}") from e


# --- Main Execution ---


def main():
    """Main entry point for the CLI."""
    # Initialize: Check config first
    try:
        models_path = get_models_root_path()
        logger.info(f"Using model registry at: {models_path}")
    except (FileNotFoundError, ValueError, NotADirectoryError) as e:
        logger.error(f"CLI initialization failed: Configuration error - {e}")
        print(f"Error: Configuration problem - {e}", file=sys.stderr)
        sys.exit(1)  # Exit if config fails

    parser = create_parser()
    args = parser.parse_args()

    # Command dispatching with centralized error handling
    try:
        if args.command == "list":
            list_models_cli()
        elif args.command == "metadata":
            show_metadata_cli(args.model_id)
        elif args.command == "train":
            if not (0 < args.test_size < 1):
                # argparse choices could also enforce this range
                raise ValueError("test_size must be between 0 and 1 (exclusive).")
            train_model_cli(
                args.model_id, args.data_source, args.target_column, args.test_size
            )
        elif args.command == "predict":
            predict_batch_cli(args.model_id, args.input, args.output)
        else:
            parser.print_help()
            sys.exit(1)

        logger.info(f"Command '{args.command}' completed successfully.")
        sys.exit(0)  # Success exit code

    # Catch specific expected errors for user-friendly messages
    except FileNotFoundError as e:
        logger.error(f"File not found during '{args.command}': {e}")
        print(f"\nError: File not found - {e}", file=sys.stderr)
        sys.exit(2)  # Specific exit code for file errors
    except (ValueError, TypeError, argparse.ArgumentTypeError) as e:
        logger.error(f"Invalid input or value during '{args.command}': {e}")
        print(f"\nError: Invalid input - {e}", file=sys.stderr)
        sys.exit(3)  # Specific exit code for value errors
    except ConnectionError as e:
        logger.error(f"Network error during '{args.command}': {e}")
        print(f"\nError: Network connection failed - {e}", file=sys.stderr)
        sys.exit(4)  # Specific exit code for network errors
    except RuntimeError as e:  # Catch our wrapped execution errors
        logger.error(
            f"Execution error during '{args.command}': {e}", exc_info=False
        )  # No need for full trace here
        print(f"\nError: Execution failed - {e}", file=sys.stderr)
        sys.exit(5)  # Specific exit code for runtime errors
    except Exception as e:  # Catch unexpected errors
        logger.exception(
            f"An unexpected error occurred during command '{args.command}': {e}"
        )
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)  # General error exit code


if __name__ == "__main__":
    main()
