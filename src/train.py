import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

from preprocess import clean_text

# Load dataset
df = pd.read_csv("../data/Stress.csv")

# Keep only required columns
df = df[['text', 'label']]

print("Dataset loaded successfully")
print(df.head())

# Preprocess text
df['processed'] = df['text'].apply(clean_text)

# Feature extraction
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['processed'])

y = df['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "../models/model.pkl")
joblib.dump(tfidf, "../models/vectorizer.pkl")

print("✅ Model trained successfully")