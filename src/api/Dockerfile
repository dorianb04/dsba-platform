# This Dockerfile is meant to be run from the root of the repo

# Base Python image (Python 3.12)
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current folder (on the host machine) that contains the whole app project into the container /app folder
COPY . /app

RUN pip install .

# Expose the port on which FastAPI will run
EXPOSE 8000

# Set environment variables
ENV DSBA_MODELS_ROOT_PATH=/app/models

# Define the default command run when starting the container: Run the FastAPI app using Uvicorn
CMD ["uvicorn", "src.api.api:app", "--host", "0.0.0.0", "--port", "8000"]
