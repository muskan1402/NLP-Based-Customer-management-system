from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from scipy.special import softmax
import os

# path relative to backend/ folder when running uvicorn from repo root:
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "roberta_model")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

LABELS = ['negative', 'neutral', 'positive']

def predict(text: str):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    out = model(**enc)
    scores = softmax(out.logits[0].detach().numpy())
    idx = int(np.argmax(scores))
    return LABELS[idx], float(scores[idx])
