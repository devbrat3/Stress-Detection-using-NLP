import pandas as pd
import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from preprocess import clean_text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(BASE_DIR, "data", "Stress.csv")
model_path = os.path.join(BASE_DIR, "models", "model.pkl")

df = pd.read_csv(data_path)
df = df[['text', 'label']].dropna()

X = df['text'].apply(clean_text)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            max_features=12000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
    ),
    (
        "model",
        LogisticRegression(
            max_iter=800,
            class_weight="balanced",
            n_jobs=-1,
            solver="lbfgs"
        )
    )
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
joblib.dump(pipeline, model_path)

print(f"Accuracy: {round(acc*100, 2)}%")
print(report)