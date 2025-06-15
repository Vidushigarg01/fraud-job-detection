import pandas as pd
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import FeatureBuilder
from src.model_training import train_xgb
import joblib

# Load data
df = load_and_clean_data('data/fake_job_postings.csv')

# Build features
builder = FeatureBuilder()
X = builder.fit_transform(df)
y = df['fraudulent']

# Train model
model = train_xgb(X, y)

# Save model & vectorizer
joblib.dump(model, 'models/final_model.pkl')
joblib.dump(builder.vectorizer, 'models/vectorizer.pkl')
print("Model retrained and saved.")
