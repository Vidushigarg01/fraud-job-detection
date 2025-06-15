from flask import Flask, request, jsonify
import joblib
import pandas as pd
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import FeatureBuilder
from src.model_training import train_xgb

app = Flask(__name__)

# Load model
model = joblib.load('models/final_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    df = load_and_clean_data(df)

    text_features = vectorizer.transform(df['description']).toarray()
    prediction = model.predict(text_features)[0]
    prob = model.predict_proba(text_features)[0][1]

    result = {
        "prediction": int(prediction),
        "fraud_probability": round(prob, 4)
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
