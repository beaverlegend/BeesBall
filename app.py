from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np

app = Flask(__name__)

# Mock function simulating our processed league-wide Statcast data engine
def calculate_bauer_leverage(count, pitch_type):
    """
    Simulates the risk management shift of a count.
    In a full build, this will query our database of millions of Statcast pitches.
    """
    if count == "0-0":
        # The Trevor Bauer Nuance: High penalty for a ball, high reward for a strike
        return {
            "current_state": "0-0 (Equal Leverage)",
            "if_strike_next_avg": 0.180,
            "if_ball_next_avg": 0.340,
            "swing_probability": 0.142, # ~1 in 7 first pitches are swung at/hit
            "risk_regime": "High Stakes Transition"
        }
    elif count == "0-1":
        return {
            "current_state": "0-1 (Pitcher Advantage)",
            "if_strike_next_avg": 0.140,
            "if_ball_next_avg": 0.220,
            "swing_probability": 0.450,
            "risk_regime": "Low Risk / Defensive Hitter"
        }
    else:
        return {
            "current_state": "1-0 (Batter Advantage)",
            "if_strike_next_avg": 0.260,
            "if_ball_next_avg": 0.390,
            "swing_probability": 0.550,
            "risk_regime": "High Risk / Aggressive Hitter"
        }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    count = data.get('count', '0-0')
    pitch = data.get('pitch_type', 'Fastball')
    
    # Calculate metrics
    metrics = calculate_bauer_leverage(count, pitch)
    return jsonify(metrics)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)