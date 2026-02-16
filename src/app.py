from fastapi import FastAPI
from pydantic import BaseModel
import os
import joblib
import torch
import numpy as np
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

app = FastAPI(title="Ticket Classification API")

# Config
BERT_MODEL_PATH = os.getenv("BERT_MODEL_PATH", "../models/bert_classifier")
ML_MODEL_PATH = os.getenv("ML_MODEL_PATH", "../models/ml_classifier/ml_model.pkl")
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH", "../models/ml_classifier/vectorizer.pkl")
MAX_LENGTH = int(os.getenv("MAX_LENGTH", 60))

CLASS_NAMES = {
    0: "Customer Service",
    1: "IT Support",
    2: "Other",
    "it support": "IT Support",
    "customer service": "Customer Service",
    "other": "Other"
}

# Request Model
class TicketRequest(BaseModel):
    text: str
    model_type: str  # "bert" or "ml"

# Load Models
@app.on_event("startup")
def load_models():
    global tokenizer_bert, model_bert, vectorizer_ml, model_ml

    tokenizer_bert = DistilBertTokenizerFast.from_pretrained(BERT_MODEL_PATH)
    model_bert = DistilBertForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
    model_bert.eval()

    vectorizer_ml = joblib.load(VECTORIZER_PATH)
    model_ml = joblib.load(ML_MODEL_PATH)

# Health Check Endpoint
@app.get("/health")
def health():
    return {"status": "ok"}

# Prediction Endpoint
@app.post("/predict")
def predict(request: TicketRequest):
    text = request.text

    if request.model_type == "bert":
        encoding = tokenizer_bert(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model_bert(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"]
            )
            probs = torch.softmax(outputs.logits, dim=1)

        pred_class = torch.argmax(probs, dim=1).item()
        confidence = float(torch.max(probs).item())

    else:
        X = vectorizer_ml.transform([text])
        pred_class = str(model_ml.predict(X)[0])

        if hasattr(model_ml, "predict_proba"):
            probs = model_ml.predict_proba(X)[0]
            confidence = float(np.max(probs))
        else:
            confidence = 1.0

    return {
        "category": CLASS_NAMES[pred_class],
        "confidence": confidence,
        "model": request.model_type
    }
