import sys
sys.path.append('./src')

from config import *
from data_preprocessing import load_and_clean_data
from feature_engineering import FeatureBuilder
from utils.alerting import send_alert_email
from utils.explainability import explain_model

import joblib
import numpy as np
import pandas as pd

# Load new data (your custom jobs to predict)
df = load_and_clean_data('data/custom_jobs.csv')  
y_true = df['fraudulent'] if 'fraudulent' in df.columns else None

# Load saved model, vectorizer & encoder
model = joblib.load('models/final_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')
ohe = joblib.load('models/ohe.pkl')

# Build features for new data using saved vectorizer and encoder
builder = FeatureBuilder(TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE)
builder.tfidf = vectorizer
builder.ohe = ohe
X = builder.transform(df)

# Predict fraud probability
probs = model.predict_proba(X)[:, 1]
y_pred = model.predict(X)

# Generate explainability plots (optional)
try:
    explain_model(model, X, vectorizer.get_feature_names_out())
except:
    print("SHAP skipped (probably too many features).")

# Trigger alerts for high-risk jobs
high_risk_idx = np.where(probs > 0.9)[0]
for idx in high_risk_idx:
    job_title = df.iloc[idx]['title']
    prob = probs[idx]
    send_alert_email(job_title, prob)

# (Optional) Print evaluation if labels exist
if y_true is not None:
    from sklearn.metrics import classification_report
    print(classification_report(y_true, y_pred))

if send_email and email_address:
    print(f"Email enabled ✅ - sending to {email_address}")
    for idx in high_risk_idx:
        job_title = df.iloc[idx]['title']
        prob = probs[idx]
        print(f"ALERT: {job_title} has prob {prob}")
        send_alert_email(job_title, prob, recipient=email_address)
        alert_count += 1
        alerted_jobs.append(f"{job_title} ({round(prob*100)}%)")

