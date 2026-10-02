# Production Dockerfile for Customer Churn MLOps Serving API
FROM python:3.10-slim

WORKDIR /app

# Environment variables to optimize Python container execution
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install runtime dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source directory and serialized trained model artifact
COPY src/ ./src/
COPY churn_model.pkl .

# Expose FastAPI application port
EXPOSE 8000

# Start Uvicorn ASGI server listening on all network interfaces
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
