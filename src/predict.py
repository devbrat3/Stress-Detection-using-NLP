import joblib
import os
import sys

# Fix path to import preprocess
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from preprocess import clean_text

# Load model and vectorizer
model = joblib.load("../models/model.pkl")
vectorizer = joblib.load("../models/vectorizer.pkl")

def predict_stress(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]

    # Convert label to readable output
    if prediction == 0:
        return "Low Stress"
    elif prediction == 1:
        return "High Stress"
    else:
        return str(prediction)