import sys
sys.path.append('src')

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Nepali Sentiment Analysis")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
print("Loading models...")
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
with open('src/classifier.pkl', 'rb') as f:
    classifier = pickle.load(f)
print("Models loaded.")

LABELS = {1: "Positive", -1: "Negative", 0: "Neutral"}
EMOJIS = {1: "😊", -1: "😔", 0: "😐"}

class TextRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "Nepali Sentiment Analysis API is running"}

@app.post("/predict")
def predict(request: TextRequest):
    """Predict sentiment of Nepali text."""
    if not request.text.strip():
        return {"error": "Text cannot be empty"}
    
    embedding = embedding_model.encode([request.text])
    prediction = classifier.predict(embedding)[0]
    probabilities = classifier.predict_proba(embedding)[0]
    confidence = float(np.max(probabilities))
    
    return {
        "text": request.text,
        "sentiment": LABELS[prediction],
        "emoji": EMOJIS[prediction],
        "confidence": round(confidence * 100, 2)
    }