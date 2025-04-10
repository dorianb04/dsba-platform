import json
import logging
import pandas as pd
from fastapi import FastAPI, HTTPException
from dsba.model_registry import list_models_ids, load_model, load_model_metadata, save_model
from dsba.model_prediction import classify_record
from dsba.model_training import run_training_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S,",
)

app = FastAPI()

# using FastAPI with defaults is very convenient
# we just add this "decorator" with the "route" we want.
# If I deploy this app on "https//mywebsite.com", this function can be called by visiting "https//mywebsite.com/models/"
@app.get("/models/")
async def list_models():
    return list_models_ids()

@app.get("/models/{model_id}/metadata")
async def metadata_getter(model_id):
    try:
        metadata = load_model_metadata(model_id)
        return metadata
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, 
            detail=f"Model with ID '{model_id}' not found"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.exception(f"Unexpected error retrieving metadata for model '{model_id}': {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


# TO DO 
@app.post("/train/")
async def train_classifier(data_source: str, target_column: str, model_id: str, test_size: float = 0.2):
    model, metadata = run_training_pipeline(data_source, target_column, model_id, test_size) 
    model_path, metadata_path = save_model(model, metadata)
    return f"model trained successfully, this is you metadata: \n {metadata} \n you can access your model here {model_path} "


@app.api_route("/predict/", methods=["GET", "POST"])
async def predict(query: str, model_id: str):
    """
    Predict the target column of a record using a model.
    The query should be a json string representing a record.
    """

    try:
        record = json.loads(query)
        model = load_model(model_id)
        metadata = load_model_metadata(model_id)
        prediction = classify_record(model, record, metadata.target_column)
        return {"prediction": prediction}
    except Exception as e:
        # We do want users to be able to see the exception message in the response
        # FastAPI will by default block the Exception and send a 500 status code
        # (In the HTTP protocol, a 500 status code just means "Internal Server Error" aka "Something went wrong but we're not going to tell you what")
        # So we raise an HTTPException that contains the same details as the original Exception and FastAPI will send to the client.
        raise HTTPException(status_code=500, detail=str(e))
