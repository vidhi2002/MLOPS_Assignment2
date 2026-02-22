FROM python:3.9-slim

# Set locale to avoid Unicode issues
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy source code
COPY src/ ./src/
COPY config.json .

# Create directories for data and models
RUN mkdir -p data/processed models

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the inference service
CMD ["uvicorn", "inference.app:app", "--host", "0.0.0.0", "--port", "8000"]
