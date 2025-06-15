
# 🚨 Fraud Job Detection System

This project helps detect fake job postings using a machine learning model. It includes a simple Gradio interface to upload job listings and optionally receive email alerts for high-risk jobs.

---

## 📌 Features

- ✅ Upload a custom `custom_jobs.csv` file to get predictions  
- ✅ Clean Gradio dashboard interface  
- ✅ Email alerts for suspicious job postings  
- ✅ Model trained on real-world dataset  
- ✅ Hosted via Hugging Face / Render (SMTP supported)

---

## 📂 Files in This Repo

| File | Description |
|------|-------------|
| `app.py` | Main Python app with Gradio UI |
| `model.pkl` | Trained machine learning model |
| `custom_jobs.csv` | Sample job listings file for testing |
| `.gitignore` | To skip large files in Git |
| `requirements.txt` | Python dependencies |
| `README.md` | You’re here! |

---

## 🧠 How It Works

1. User uploads a `.csv` file with job listings  
2. The model analyzes each job and labels it as **Legitimate** or **Fraudulent**  
3. If email alerts are enabled, suspicious jobs trigger an **alert email**

---

## 📧 Email Alert Setup

- SMTP is used to send alerts via email.
- You can configure sender credentials (like Gmail or SendGrid) in `app.py`.

> **Note**: Email sending works only on platforms that support outbound SMTP (like Render, not Hugging Face Spaces).

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/<your-username>/fraud-job-detector.git
cd fraud-job-detector
pip install -r requirements.txt
python app.py
```

---

## 📊 Custom CSV Format

Make sure your `custom_jobs.csv` has columns like:

- `title`, `location`, `company_profile`, `description`, etc.

This format should match the training dataset.

---

## 🌐 Live Demo

> 🔗 https://fraud-job-detection-2.onrender.com/

---




