import json
import pytest
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient

# Adjust import based on your actual project structure
from api.api import app

# Create a test client
client = TestClient(app)

# Tests for listing models endpoint
def test_list_models():
    with patch("api.api.list_models_ids") as mock_list_models:
        # Setup mock
        mock_list_models.return_value = ["model_1", "model_2"]
        
        # Make request
        response = client.get("/models/")
        
        # Assertions
        assert response.status_code == 200
        assert response.json() == ["model_1", "model_2"]
        mock_list_models.assert_called_once()

# Tests for metadata endpoint
def test_metadata_getter():
    metadata = {"model_id": "test_model", "accuracy": 0.85}
    
    with patch("api.api.load_model_metadata") as mock_load_metadata:
        mock_load_metadata.return_value = metadata
        
        # Make request
        response = client.get("/models/test_model/metadata")
        
        # Assertions
        assert response.status_code == 200
        assert response.json() == metadata

# Tests for training endpoint
def test_train_classifier():
    """Test the classifier training endpoint."""
    # Simple 2D features and targets
    train_data = {
        "features": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
        "target": [0, 1, 0, 1],
        "model_id": "test_model_1" 
    }
    
    # Enable debugging
    import sys
    # Increase recursion limit for debugging only
    sys.setrecursionlimit(2000)  # Only use during testing
    
    # Add debugging information
    print("Sending POST request to /train endpoint...")
    
    try:
        response = client.post("/train", json=train_data)
        print(f"Response received: {response.status_code}")
        print(f"Response content: {response.text[:200]}...")  # Show first 200 chars
        assert response.status_code == 200
        assert "model_id" in response.json()
    except RecursionError as e:
        # This helps identify where the recursion is happening
        import traceback
        print(f"RecursionError. Traceback:\n{traceback.format_exc()}")
        assert False, "RecursionError detected - check for circular imports or function calls"
    except Exception as e:
        print(f"Other exception: {str(e)}")
        assert False, f"Test failed with exception: {str(e)}"


# Tests for prediction endpoint
def test_predict():
    with patch("api.api.load_model") as mock_load_model, \
         patch("api.api.load_model_metadata") as mock_load_metadata, \
         patch("api.api.classify_record") as mock_classify:
        
        # Setup mocks
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        
        # Create a simple metadata object with just what's needed
        metadata = type('obj', (object,), {"target_column": "target"})
        mock_load_metadata.return_value = metadata
        
        mock_classify.return_value = 1
        
        # Make request
        sample_record = {"feature1": 1, "feature2": 2}
        query = json.dumps(sample_record)
        response = client.post("/predict/", params={"query": query, "model_id": "test_model"})
        
        # Assertions
        assert response.status_code == 200
        assert response.json()["prediction"] == 1

def test_simple():
    assert True

def test_predict_error_handling():
    with patch("api.api.load_model") as mock_load_model:
        # Setup mock to raise an exception
        mock_load_model.side_effect = ValueError("Test error")
        
        # Make request
        query = json.dumps({"feature1": 1})
        response = client.post("/predict/", params={"query": query, "model_id": "test_model"})
        
        # Assertions
        assert response.status_code == 500
        assert "Test error" in response.json()["detail"]
