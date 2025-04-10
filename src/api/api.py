import logging
from typing import Any

from dsba.model_prediction import classify_record
from dsba.model_registry import (
    ClassifierMetadata,
    list_models_ids,
    load_model,
    load_model_metadata,
    save_model,
)
from dsba.model_training import run_training_pipeline
from fastapi import (
    FastAPI,
    HTTPException,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
)

# Basic Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Pydantic Models ---
# Define structured request and response models


class TrainRequest(BaseModel):
    model_id: str = Field(
        ...,
        description="Unique identifier for the model to be trained.",
        json_schema_extra={"examples": ["titanic_classifier_v2"]},
    )
    data_source: HttpUrl | str = Field(
        ...,
        description="URL (http/https) or local path to the CSV training data.",
        json_schema_extra={
            "examples": [
                "tests/data/sample_training_data.csv",
                "https://example.com/data.csv",
            ]
        },
    )
    target_column: str = Field(
        ...,
        description="Name of the target variable column.",
        json_schema_extra={"examples": ["target"]},
    )
    test_size: float | None = Field(
        0.2, gt=0, lt=1, description="Proportion of data for testing (0 < size < 1)."
    )


class PredictRequest(BaseModel):
    model_id: str = Field(
        ...,
        description="ID of the trained model for prediction.",
        json_schema_extra={"examples": ["titanic_classifier_v2"]},
    )
    features: dict[str, Any] = Field(
        ...,
        description="Dictionary of feature names and values for prediction.",
        json_schema_extra={
            "examples": [{"feature1": 2.5, "feature2": "A", "feature3": 3.1}]
        },
    )


class PredictResponse(BaseModel):
    model_id: str
    prediction: Any  # Allow various prediction types


# Define response model mirroring ClassifierMetadata
class ModelMetadataResponse(BaseModel):
    id: str
    target_column: str
    algorithm: str
    created_at: str
    hyperparameters: dict[str, Any]
    description: str
    performance_metrics: dict[
        str, float | str
    ]  # Allow str for potential error messages
    model_config = ConfigDict(
        from_attributes=True,
    )


# --- FastAPI App ---

app = FastAPI(
    title="DSBA MLOps Platform API",
    description="API for training, evaluating, and predicting with models.",
    version="1.0.0",
)

# --- API Endpoints ---


@app.get("/", tags=["General"])
async def read_root():
    """Root endpoint providing a welcome message."""
    logger.info("Root endpoint accessed.")
    return {"message": "Welcome to the DSBA MLOps Platform API!"}


@app.get("/models/", response_model=list[str], tags=["Models"])
async def list_available_models_api():
    """Lists the IDs of all models available in the registry."""
    logger.info("Request received for listing models.")
    try:
        model_ids = list_models_ids()
        return model_ids
    except Exception as e:
        logger.exception("Error listing models.")
        raise HTTPException(
            status_code=500, detail=f"Internal server error listing models: {e!s}"
        ) from e


@app.get(
    "/models/{model_id}/metadata",
    response_model=ModelMetadataResponse,
    tags=["Models"],
    responses={
        404: {"description": "Model not found"},
        400: {"description": "Invalid metadata format"},
        500: {"description": "Internal Server Error"},
    },
)
async def get_model_metadata_api(model_id: str):
    """Retrieves the metadata for a specific model ID."""
    logger.info(f"Request received for metadata of model: {model_id}")
    try:
        # Load metadata (which is already a dataclass)
        metadata: ClassifierMetadata = load_model_metadata(model_id)
        # Pydantic model with orm_mode=True handles conversion
        return metadata
    except FileNotFoundError as e:
        logger.warning(f"Metadata not found for model ID: {model_id}")
        raise HTTPException(
            status_code=404, detail=f"Model metadata not found for ID: {model_id}"
        ) from e
    except ValueError as e:
        logger.error(f"Error loading or parsing metadata for model {model_id}: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid metadata format or content for model {model_id}: {e!s}",
        ) from e
    except Exception as e:
        logger.exception(
            f"Unexpected error retrieving metadata for model '{model_id}': {e}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error retrieving metadata: {e!s}",
        ) from e


@app.post(
    "/train/",
    # Use the Pydantic model for the response
    response_model=ModelMetadataResponse,
    status_code=201,  # 201 Created is appropriate
    tags=["Training"],
    responses={
        400: {"description": "Bad Request (e.g., invalid input, data error)"},
        500: {"description": "Internal Server Error (training failed)"},
    },
)
# Use the Pydantic model for the request body
async def train_model_api(request: TrainRequest):
    """
    Trigger model training & evaluation using data from URL/path.
    Saves model and metadata. Returns metadata.
    """
    logger.info(f"Received training request for model_id: {request.model_id}")
    # Convert potential HttpUrl back to string for the pipeline function
    data_source_str = str(request.data_source)

    try:
        # Call the training pipeline function
        model, metadata = run_training_pipeline(
            data_source=data_source_str,
            target_column=request.target_column,
            model_id=request.model_id,
            test_size=request.test_size,
        )

        # Save the trained model and its metadata
        save_model(model, metadata)
        logger.info(f"Successfully trained and saved model: {request.model_id}")

        # Return the structured metadata object (FastAPI handles serialization)
        return metadata

    except FileNotFoundError as e:
        logger.error(f"Training data not found: {data_source_str}. Error: {e}")
        raise HTTPException(
            status_code=400, detail=f"Training data file not found: {e!s}"
        ) from e
    except ConnectionError as e:
        logger.error(
            f"Could not access training data URL: {data_source_str}. Error: {e}"
        )
        raise HTTPException(
            status_code=400, detail=f"Could not access training data URL: {e!s}"
        ) from e
    except ValueError as e:
        logger.error(f"Data or value error during training for {request.model_id}: {e}")
        raise HTTPException(
            status_code=400, detail=f"Error during training data processing: {e!s}"
        ) from e
    except RuntimeError as e:
        logger.error(
            f"Runtime error during training execution for {request.model_id}: {e}"
        )
        raise HTTPException(
            status_code=500, detail=f"Model training execution failed: {e!s}"
        ) from e
    except Exception as e:
        logger.exception(
            f"Unexpected error during training for model {request.model_id}."
        )
        raise HTTPException(
            status_code=500, detail=f"Internal server error during training: {e!s}"
        ) from e


# Define separate GET/POST if needed, or keep combined if simple
# Using POST is generally better for sending data (features)
@app.post(
    "/predict/",
    response_model=PredictResponse,  # Use Pydantic response model
    tags=["Prediction"],
    responses={
        404: {"description": "Model not found"},
        400: {"description": "Invalid input features"},
        500: {"description": "Prediction failed"},
    },
)
# Use Pydantic model for request body
async def predict_api(request: PredictRequest):
    """
    Makes a prediction for a single data record using a specified model ID.
    Input features should be provided as a dictionary in the JSON request body.
    """
    logger.info(f"Received prediction request for model_id: {request.model_id}")
    try:
        # Load model and metadata
        model = load_model(request.model_id)
        # Metadata needed only for target column name if classify_record requires it
        # It might be more efficient if classify_record doesn't need metadata
        metadata = load_model_metadata(request.model_id)
        prediction = classify_record(model, request.features, metadata.target_column)
        logger.info(
            f"Prediction successful for model {request.model_id}. Result: {prediction}"
        )

        return PredictResponse(model_id=request.model_id, prediction=prediction)

    except FileNotFoundError as e:
        logger.warning(
            f"Model or metadata not found for prediction: {request.model_id}"
        )
        raise HTTPException(
            status_code=404,
            detail=f"Model (or its metadata) not found for ID: {request.model_id}",
        ) from e
    except (ValueError, TypeError) as e:
        logger.error(f"Prediction input/value error for {request.model_id}: {e}")
        raise HTTPException(
            status_code=400, detail=f"Prediction input data error: {e!s}"
        ) from e
    except Exception as e:
        logger.exception(
            f"Unexpected error during prediction for model {request.model_id}."
        )
        raise HTTPException(
            status_code=500, detail=f"Internal server error during prediction: {e!s}"
        ) from e
