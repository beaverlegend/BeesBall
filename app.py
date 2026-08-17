from flask import Flask, request, jsonify
import joblib
from src.evaluator import calculate_optimal_approach

app = Flask(__name__)

# Load models and encoder
clf = joblib.load('pitch_clf.pkl')
reg = joblib.load('run_val_reg.pkl')
encoder = joblib.load('pitch_encoder.pkl')

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json or {}
    
    # Parse inputs with robust defaults
    p_throws = data.get('p_throws', 'R').upper()
    stand = data.get('stand', 'R').upper()
    
    input_params = {
        'balls': int(data.get('balls', 0)),
        'strikes': int(data.get('strikes', 0)),
        'outs': int(data.get('outs', 0)),
        'inning': int(data.get('inning', 1)),
        'same_handedness': 1 if p_throws == stand else 0
    }
    
    evaluation = calculate_optimal_approach(clf, reg, encoder, input_params)
    
    return jsonify({
        'status': 'success',
        'matchup_context': {
            'pitcher_hand': p_throws,
            'batter_hand': stand,
            'count': f"{input_params['balls']}-{input_params['strikes']}"
        },
        'results': evaluation
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)