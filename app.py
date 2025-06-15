import gradio as gr
import pandas as pd
import numpy as np
import joblib
import sys
import os
sys.path.append('./src')

from config import *
from data_preprocessing import load_and_clean_data
from feature_engineering import FeatureBuilder
from utils.alerting import send_alert_email
from utils.explainability import explain_model
from sklearn.metrics import classification_report

# Load saved models
model = joblib.load('models/final_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')
ohe = joblib.load('models/ohe.pkl')

# Feature builder
builder = FeatureBuilder(TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE)
builder.tfidf = vectorizer
builder.ohe = ohe

def predict_and_alert(file, send_email, email_address):
    try:
        df = load_and_clean_data(file.name)
        y_true = df['fraudulent'] if 'fraudulent' in df.columns else None

        X = builder.transform(df)
        probs = model.predict_proba(X)[:, 1]
        y_pred = model.predict(X)

        df['Fraud Probability'] = np.round(probs, 3)
        df['Predicted Label'] = y_pred

        high_risk_idx = np.where(probs > 0.5)[0]  # lowered threshold

        alert_count = 0
        alerted_jobs = []

        print(f"\n⚙️ Email Option: {send_email}, Address: {email_address}")
        print(f"📌 High Risk Jobs Indexes: {high_risk_idx}")

        if send_email and email_address:
            for idx in high_risk_idx:
                job_title = df.iloc[idx]['title']
                prob = probs[idx]

                print(f"📤 Sending alert for '{job_title}' with prob {prob}")
                try:
                    send_alert_email(job_title, prob, recipient=email_address)
                    alert_count += 1
                    alerted_jobs.append(f"{job_title} ({round(prob*100)}%)")
                except Exception as e:
                    print(f"❌ Email failed: {e}")

        report = ""
        if y_true is not None:
            report = classification_report(y_true, y_pred)

        if send_email:
            msg = f"{alert_count} alert email(s) sent to {email_address}.\n" + "\n".join(alerted_jobs) if alert_count else "No alerts sent (all probs < 50%)."
        else:
            msg = "Email alerts were disabled."

        return df[['title', 'location', 'Fraud Probability', 'Predicted Label']], report + "\n\n" + msg

    except Exception as e:
        return None, f"❌ Error: {str(e)}"


# Gradio UI
demo = gr.Interface(
    fn=predict_and_alert,
    inputs=[
        gr.File(label="Upload custom_jobs.csv", file_types=['.csv']),
        gr.Checkbox(label="Send Email Alerts?", value=True),
        gr.Textbox(label="Enter Your Email Address", placeholder="example@email.com")
    ],
    outputs=[
        gr.Dataframe(label="Prediction Results"),
        gr.Textbox(label="Report & Alerts")
    ],
    title="Fake Job Detector with Email Alerts",
    description="Upload a job CSV to detect fake postings. Optionally send alert emails for high-risk jobs."
)
demo.launch(
    server_name="0.0.0.0", 
    server_port=int(os.environ.get("PORT", 7860))  # Render sets the PORT env variable
)

if __name__ == "__main__":
    demo.launch(share=True)
