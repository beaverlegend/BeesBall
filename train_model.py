import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. Load cached Statcast data
df = pd.read_parquet('statcast_2025.parquet')

# 2. Features (X) & Target (y)
X = df[['balls', 'strikes', 'outs_when_up', 'inning']]
y = df['pitch_type']  # Target: Predict pitch type

# 3. Train
model = RandomForestClassifier()
model.fit(X, y)

# 4. Save model to disk
joblib.dump(model, 'pitch_model.pkl')