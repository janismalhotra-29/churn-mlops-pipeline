# Production Dockerfile for Customer Churn MLOps Serving API
FROM python:3.10-slim

WORKDIR /app

# Environment variables to optimize Python container execution
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install runtime dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source directory, data files, serialized model artifact, and database
COPY src/ ./src/
COPY data/ ./data/
COPY churn_model.pkl .
COPY mlflow.db .

# Expose FastAPI application port
EXPOSE 8000

# Start Uvicorn ASGI server listening on Render's dynamic PORT or default 8000
CMD ["sh", "-c", "uvicorn src.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
