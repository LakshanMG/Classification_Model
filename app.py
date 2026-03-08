from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import pickle
import numpy as np
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

app = FastAPI()


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'programming_classifier_model.h5'))

with open(os.path.join(BASE_DIR, 'tokenizer.pkl'), 'rb') as f:
    tokenizer = pickle.load(f)

with open(os.path.join(BASE_DIR, 'label_encoder.pkl'), 'rb') as f:
    label_encoder = pickle.load(f)

class ProblemRequest(BaseModel):
    problem: str

@app.post("/predict")
async def predict_complexity(request: ProblemRequest):
    text = request.problem
    clean_text = re.sub(r'[^a-zA-Z\s]', '', text).lower().strip()
    
    seq = tokenizer.texts_to_sequences([clean_text])
    padded = pad_sequences(seq, maxlen=200, padding='post', truncating='post')
    
    prediction = model.predict(padded, verbose=0)
    class_idx = np.argmax(prediction, axis=1)[0]
    confidence = float(np.max(prediction) * 100)
    complexity = str(label_encoder.inverse_transform([class_idx])[0])
    
    return {
        "complexity": complexity,
        "confidence": round(confidence, 2)
    }

@app.get("/")
def home():
    return {"status": "Classifier API is running"}