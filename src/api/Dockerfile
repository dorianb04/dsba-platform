# Base Python image (Python 3.12)
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy all files into container
COPY . /app

# Install your package (assuming pyproject.toml exists in root and project uses PEP 517/518 standards)
RUN pip install .

# Expose FastAPI port
EXPOSE 8000

# Environment variable (optional, but okay!)
ENV DSBA_MODELS_ROOT_PATH=/app/models

# Run FastAPI using Uvicorn
CMD ["uvicorn", "src.api.api:app", "--host", "0.0.0.0", "--port", "8000"]
