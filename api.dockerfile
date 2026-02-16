FROM python:3.11-slim

WORKDIR /app

# System dependencies (optional but safe)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install API requirements
COPY requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

# Copy source code
COPY src/app.py .

# Copy models
COPY models /models

# Environment variables for model paths
ENV BERT_MODEL_PATH=/models/bert_classifier
ENV ML_MODEL_PATH=/models/ml_classifier/ml_model.pkl
ENV VECTORIZER_PATH=/models/ml_classifier/vectorizer.pkl
ENV MAX_LENGTH=60

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
