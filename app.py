from flask import Flask, jsonify, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained ML model at server startup
model = joblib.load('pitch_model.pkl')


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json

    # Convert request into DataFrame matching model input features
    input_data = pd.DataFrame([
        {
            'balls': int(data.get('balls', 0)),
            'strikes': int(data.get('strikes', 0)),
            'outs_when_up': int(data.get('outs', 0)),
            'inning': int(data.get('inning', 1)),
        }
    ])

    # Predict probabilities for each pitch type
    probabilities = model.predict_proba(input_data)[0]
    classes = model.classes_

    # Format result map (e.g., {'FF': 0.55, 'SL': 0.30, 'CH': 0.15})
    predictions = {
        cls: round(float(prob), 3) for cls, prob in zip(classes, probabilities)
    }

    return jsonify({'predicted_pitch_probabilities': predictions})