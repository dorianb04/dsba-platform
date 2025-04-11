FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

COPY pyproject.toml README.md ./
COPY ./src ./src
RUN mkdir -p /app/models/registry

# Install dependencies AND the project itself from the current directory (.) in the container
RUN pip install --no-cache-dir .

# Expose the port the app runs on port 8000
EXPOSE 8000

# Define the command to run the application using Uvicorn
CMD ["uvicorn", "src.api.api:app", "--host", "0.0.0.0", "--port", "8000"]
