import joblib
import os
import sys

# ✅ Get absolute base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ✅ Add src to path (for preprocess import)
sys.path.append(os.path.join(BASE_DIR, "src"))

from preprocess import clean_text

# ✅ Safe model paths (works locally + Streamlit cloud)
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

# ✅ Load model safely
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
except Exception as e:
    raise RuntimeError(f"❌ Error loading model files: {e}")

# ✅ Prediction function
def predict_stress(text):
    try:
        cleaned = clean_text(text)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        # Convert label → readable output
        if prediction == 0:
            return "Low Stress"
        elif prediction == 1:
            return "High Stress"
        else:
            return f"Unknown ({prediction})"

    except Exception as e:
        return f"Prediction Error: {e}"