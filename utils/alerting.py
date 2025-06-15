import smtplib
from email.message import EmailMessage

def send_alert_email(job_title, prob, recipient="gargvidushi06@gmail.com"):
    sender = "fraudjobdetector@gmail.com"
    password = "fssp snax soyh ueix"

    msg = EmailMessage()
    msg['Subject'] = "High Risk Fraudulent Job Detected!"
    msg['From'] = sender
    msg['To'] = recipient

    body = f"ALERT: Job '{job_title}' predicted with fraud probability {prob*100:.2f}%"
    msg.set_content(body)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(sender, password)
        smtp.send_message(msg)

    print("Alert email sent.")
