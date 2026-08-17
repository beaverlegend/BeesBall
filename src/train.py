import joblib
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def train_models(data_path="data/statcast_2025.parquet"):
    df = pd.read_parquet(data_path)
    
    # Define features
    features = [
        'balls', 'strikes', 'outs_when_up', 'inning', 
        'same_handedness', 'release_speed', 'pfx_x', 'pfx_z'
    ]
    
    # Encode target pitch types (FF, SL, CH, etc.)
    pitch_encoder = LabelEncoder()
    df['pitch_type_encoded'] = pitch_encoder.fit_transform(df['pitch_type'])
    
    X = df[features]
    y_pitch = df['pitch_type_encoded']
    y_run_value = df['delta_run_exp'].fillna(0.0)
    
    # Train Pitch Selection Classifier
    print("Training Pitch Selection Model...")
    clf = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    clf.fit(X, y_pitch)
    
    # Train Run Value Expectation Regressor
    print("Training Run Value Evaluator...")
    reg = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    reg.fit(X, y_run_value)
    
    # Save models and encoder
    joblib.dump(clf, 'pitch_clf.pkl')
    joblib.dump(reg, 'run_val_reg.pkl')
    joblib.dump(pitch_encoder, 'pitch_encoder.pkl')
    print("Models saved successfully.")

if __name__ == "__main__":
    train_models()