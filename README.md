# DSBA MLOps Platform

A minimal MLOps platform for educational purposes, demonstrating model training, prediction via API and CLI, containerization with Docker, and deployment to Azure Container Instances (ACI) using Azure CLI.

## Project Goal

This project provides a basic framework to:
* Train simple classification models using data from local files or URLs.
* Register models and their metadata locally.
* Serve prediction requests via a FastAPI API.
* Interact with the platform via a command-line interface (CLI).
* Package the API into a Docker container.
* Deploy the containerized API to Azure Container Instances with persistent model storage using Azure CLI.

## Project Structure

* `pyproject.toml`: Defines project metadata, dependencies (main and dev), and build system (hatchling).
* `.env.example`: Template showing required environment variables for local development.
* `src/`: Contains all source code.
    * `api/`: FastAPI application (`api.py`) and its `Dockerfile`.
    * `cli/`: Command-line interface (`main.py`) and its `__init__.py`.
    * `dsba/`: Core library code (data ingestion, model registry, training, prediction, config, etc.).
    * `notebooks/`: Example Jupyter notebooks (e.g., `model_training_example.ipynb`).
* `tests/`: Contains pytest unit and integration tests.
    * `data/`: Sample data for testing.
* `.gitignore`: Specifies files and directories to be excluded from Git.
* `.pre-commit-config.yaml`: Configuration for pre-commit hooks (linting, formatting, testing).

## Prerequisites

Ensure the following software is installed on your system:

1.  **Python:** Version 3.12 ([python.org](https://www.python.org)).
2.  **Git:** ([git-scm.com](https://git-scm.com)).
3.  **Docker Desktop:** Install and ensure the Docker engine is running ([docker.com](https://www.docker.com/products/docker-desktop/)).
4.  **Azure CLI:** Install and login ([docs.microsoft.com](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli)). Run `az login`.

## Step 1: Local Setup

1.  **Fork & Clone:** Fork this repository on GitHub, then clone your fork:
    ```bash
    git clone https://github.com/dorianb04/dsba-platform.git
    cd dsba-platform
    ```
2.  **Create `develop` Branch (Optional but Recommended):**
    ```bash
    git checkout main
    git checkout -b develop
    # git push -u origin develop # If you want to push it immediately
    ```
3.  **Set Up Python Virtual Environment:**
    ```powershell
    # Using PowerShell in the project root
    python -m venv .venv
    .\.venv\Scripts\activate
    ```
    *(Use `source .venv/bin/activate` on macOS/Linux)*
4.  **Install Dependencies:**
    ```powershell
    python -m pip install --upgrade pip
    pip install -e ".[dev]"
    ```
    *(This installs the project in editable mode plus dev dependencies like pytest, ruff, python-dotenv)*
5.  **Set up Pre-commit Hooks (Optional but Recommended):**
    ```powershell
    pip install pre-commit
    pre-commit install
    ```

## Step 2: Local Configuration

This project uses environment variables for configuration, primarily for the model registry path. For local development, use a `.env` file.

1.  **Create `.env` File:** In the project root, create a file named `.env`.
2.  **Add Environment Variable:** Add the path to your desired *local* model storage directory. Create this directory if it doesn't exist.
    ```dotenv
    # Content of .env file
    # Use absolute path or a path relative to the project root
    DSBA_MODELS_ROOT_PATH=models_registry
    ```
3.  **Ensure `.gitignore`:** Verify that `.env` is listed in your `.gitignore` file.

*Note: The `src/dsba/config.py` script uses `python-dotenv` to automatically load variables from the `.env` file when running locally.*

## Step 3: Code Setup

Ensure your codebase includes the necessary updates and refactoring applied during development:

* `src/dsba/config.py`: Relies only on the `DSBA_MODELS_ROOT_PATH` environment variable.
* `src/cli/dsba_cli` is renamed to `src/cli/main.py`.
* `src/cli/__init__.py` exists (can be empty).
* `src/dsba/model_training.py`: Uses helper functions and calls `data_ingestion` correctly.
* `src/api/api.py`: Uses Pydantic models and has proper error handling.
* `src/api/Dockerfile`: Copies source files correctly before `pip install`.

## Step 4: Local Usage Verification

Ensure your virtual environment is active and the `DSBA_MODELS_ROOT_PATH` is set in your `.env` file.

1.  **Run CLI Commands:** (Execute from project root)
    ```powershell
    # List models
    python src/cli/main.py list

    # Train a sample model
    python src/cli/main.py train --model-id local_sample_v1 --data-source tests/data/sample_training_data.csv --target-column target

    # List models again
    python src/cli/main.py list

    # Get metadata
    python src/cli/main.py metadata local_sample_v1

    # Predict using the trained model
    python src/cli/main.py predict --model-id local_sample_v1 --input tests/data/sample_training_data.csv --output local_predictions.csv
    ```
    *Check the console output and the creation of files in your `models_registry` directory and `local_predictions.csv`.*

2.  **Run API Locally:**
    ```powershell
    uvicorn src.api.api:app --reload --host 127.0.0.1 --port 8000
    ```
    *Open `http://127.0.0.1:8000/docs` in your browser to interact with the API via Swagger UI.*
    *Stop the server with `CTRL+C`.*

## Step 5: Dockerize the API

1.  **Ensure Docker Desktop is running.**
2.  **Build the Image:** (Run from project root)
    ```powershell
    docker build -t dsba-platform-api:latest -f src/api/Dockerfile .
    ```

## Step 6: Prepare for Azure Deployment

1.  **Set up Azure Container Registry (ACR):**
    * Choose a **globally unique name** for your ACR (e.g., `dsbamlops<random_number>`).
    * Define variables (example using PowerShell):
      ```powershell
      $RESOURCE_GROUP="dsba-mlops-rg-france" # Or your preferred RG name
      $LOCATION="France Central"
      $ACR_NAME="<Your-Unique-ACR-Name>" # Replace with your chosen unique name
      $IMAGE_TAG="v1.0.0" # Choose a version tag for your image
      ```
    * Create Resource Group (if it doesn't exist):
      ```powershell
      az group create --name $RESOURCE_GROUP --location $LOCATION
      ```
    * Create ACR:
      ```powershell
      az acr create --name $ACR_NAME --resource-group $RESOURCE_GROUP --sku Basic --admin-enabled true
      ```
      *Note the ACR Login Server: `$($ACR_NAME).azurecr.io`*
2.  **Push Image to ACR:**
    ```powershell
    # Define full image path
    $ACR_LOGIN_SERVER="$($ACR_NAME).azurecr.io"
    $ACR_IMAGE_NAME="$($ACR_LOGIN_SERVER)/samples/dsba-platform-api:$IMAGE_TAG"

    # Login to ACR
    az acr login --name $ACR_NAME

    # Tag the locally built image
    docker tag dsba-platform-api:latest $ACR_IMAGE_NAME

    # Push the image
    docker push $ACR_IMAGE_NAME
    ```

## Step 7: Azure Deployment via CLI

This will deploy the container image to Azure Container Instances (ACI) with persistent storage for models using Azure Files.

1.  **Define Deployment Variables:**
    ```powershell
    # Use the $RESOURCE_GROUP, $LOCATION, $ACR_NAME defined previously
    $ACI_NAME="dsba-api-instance-$(Get-Random -Maximum 1000)"       # Unique name for ACI
    $STORAGE_ACCOUNT_NAME="dsbastorage$(Get-Random)"               # Globally unique storage name
    $MODELS_SHARE_NAME="modelshare"                                # File share ONLY for models
    $DNS_NAME_LABEL="dsba-mlops-api-$(Get-Random -Maximum 1000)"   # Globally unique DNS label
    $MODELS_MOUNT_PATH_IN_CONTAINER="/app/models/registry"         # ABSOLUTE path inside container
    $ACR_IMAGE_NAME_WITH_TAG=$ACR_IMAGE_NAME                       # Full image path from previous step
    ```

2.  **Create Storage Account:**
    ```powershell
    Write-Host "Creating storage account: $STORAGE_ACCOUNT_NAME..."
    az storage account create --name $STORAGE_ACCOUNT_NAME --resource-group $RESOURCE_GROUP --location $LOCATION --sku Standard_LRS --kind StorageV2
    ```

3.  **Get Storage Account Key:** *(Handle Securely!)*
    ```powershell
    Write-Host "Retrieving storage account key..."
    $STORAGE_KEY = (az storage account keys list --resource-group $RESOURCE_GROUP --account-name $STORAGE_ACCOUNT_NAME --query "[0].value" --output tsv)
    if (-not $STORAGE_KEY) { Write-Error "Failed to retrieve storage key!"; return }
    Write-Host "Storage Key obtained."
    ```

4.  **Create Models File Share:**
    ```powershell
    Write-Host "Creating models file share: $MODELS_SHARE_NAME..."
    az storage share create --name $MODELS_SHARE_NAME --account-name $STORAGE_ACCOUNT_NAME --account-key $STORAGE_KEY --quota 5
    ```

5.  **Get ACR Credentials:**
    ```powershell
    Write-Host "Getting ACR Admin Credentials for ACI..."
    $ACR_USERNAME=$ACR_NAME
    $ACR_PASSWORD = (az acr credential show --name $ACR_NAME --query passwords[0].value --output tsv)
    if (-not $ACR_PASSWORD) { Write-Error "Failed to retrieve ACR password!"; return }
    Write-Host "ACR credentials for ACI obtained."
    ```

6.  **Create Azure Container Instance:**
    ```powershell
    Write-Host "Creating Azure Container Instance: $ACI_NAME..."
    az container create `
      --resource-group $RESOURCE_GROUP `
      --name $ACI_NAME `
      --image $ACR_IMAGE_NAME_WITH_TAG `
      --dns-name-label $DNS_NAME_LABEL `
      --ports 8000 `
      --cpu 1 `
      --memory 3 `
      --os-type Linux `
      --location $LOCATION `
      --environment-variables "PYTHONUNBUFFERED=1" "DSBA_MODELS_ROOT_PATH=$MODELS_MOUNT_PATH_IN_CONTAINER" `
      --registry-login-server $ACR_LOGIN_SERVER `
      --registry-username $ACR_USERNAME `
      --registry-password $ACR_PASSWORD `
      --azure-file-volume-account-name $STORAGE_ACCOUNT_NAME `
      --azure-file-volume-account-key $STORAGE_KEY `
      --azure-file-volume-share-name $MODELS_SHARE_NAME `
      --azure-file-volume-mount-path $MODELS_MOUNT_PATH_IN_CONTAINER
    ```
    *Wait for the deployment to complete.*

## Step 8: Verification

1.  **Get FQDN:**
    ```powershell
    Write-Host "Getting container FQDN..."
    az container wait --resource-group $RESOURCE_GROUP --name $ACI_NAME --created
    $FQDN = (az container show --resource-group $RESOURCE_GROUP --name $ACI_NAME --query "ipAddress.fqdn" --output tsv)
    if ($FQDN) {
        $API_URL="http://$($FQDN):8000"
        Write-Host "-----------------------------------------------------" -ForegroundColor Green
        Write-Host "Deployment Successful!" -ForegroundColor Green
        Write-Host "API URL: $API_URL"
        Write-Host "Swagger UI: $API_URL/docs"
        Write-Host "-----------------------------------------------------" -ForegroundColor Green
    } else { Write-Error "Could not retrieve FQDN." }
    ```
2.  **Test API:** Open the **Swagger UI URL** in your browser. Use the interface or tools like `curl`/Python client to interact with your deployed API endpoints. Train a model via the API and verify it appears in your Azure File Share (`modelshare`).

## Step 9: Clean Up (Optional)

* When finished, delete the Azure resources to avoid ongoing costs:
    ```powershell
    Write-Host "Deleting resource group $RESOURCE_GROUP and all its resources..."
    az group delete --name $RESOURCE_GROUP --yes --no-wait
    ```

## Testing

This project includes unit and integration tests using `pytest`. Ensure development dependencies are installed (`pip install -e ".[dev]"`) and run tests from the project root:

```bash
pytest
```
