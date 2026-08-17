import pandas as pd
import numpy as np

def calculate_optimal_approach(clf, reg, encoder, input_dict):
    """
    Evaluates pitch options and determines the pitch type with the lowest Expected Run Value.
    """
    classes = encoder.classes_
    results = []
    
    # Baseline movement profiles per pitch type (league average placeholders)
    pitch_profiles = {
        # Fastballs
        'FF': {'speed': 94.2, 'pfx_x': -0.5, 'pfx_z': 1.3},  # Four-Seam Fastball
        'SI': {'speed': 93.3, 'pfx_x': -1.2, 'pfx_z': 0.6},  # Sinker / Two-Seam
        'FC': {'speed': 89.1, 'pfx_x': 0.2,  'pfx_z': 0.6},  # Cutter
        'FA': {'speed': 92.0, 'pfx_x': -0.6, 'pfx_z': 1.0},  # Fastball (Generic)

        # Offspeed / Breaking
        'SL': {'speed': 84.8, 'pfx_x': 0.6,  'pfx_z': -0.1}, # Slider
        'ST': {'speed': 81.5, 'pfx_x': 1.2,  'pfx_z': -0.2}, # Sweeper
        'SV': {'speed': 82.0, 'pfx_x': 0.8,  'pfx_z': -0.5}, # Slurve
        'CH': {'speed': 85.1, 'pfx_x': -1.1, 'pfx_z': 0.4},  # Changeup
        'CU': {'speed': 78.5, 'pfx_x': 0.8,  'pfx_z': -1.0}, # Curveball
        'KC': {'speed': 81.0, 'pfx_x': 0.9,  'pfx_z': -1.1}, # Knuckle Curve
        'CS': {'speed': 72.0, 'pfx_x': 1.1,  'pfx_z': -1.3}, # Slow Curve / Eephus

        # Specialty / Rare
        'FS': {'speed': 86.0, 'pfx_x': -0.8, 'pfx_z': 0.2},  # Splitter
        'FO': {'speed': 84.0, 'pfx_x': -0.9, 'pfx_z': 0.1},  # Forkball
        'KN': {'speed': 76.0, 'pfx_x': 0.1,  'pfx_z': 0.1},  # Knuckleball
        'SC': {'speed': 82.0, 'pfx_x': 1.0,  'pfx_z': 0.3},  # Screwball
        'EP': {'speed': 65.0, 'pfx_x': -0.2, 'pfx_z': -0.5}, # Eephus
    }
    
    # Create feature vectors for candidate pitch types
    candidate_rows = []
    valid_classes = []
    
    for cls in classes:
        profile = pitch_profiles.get(cls, {'speed': 90.0, 'pfx_x': 0.0, 'pfx_z': 0.0})
        candidate_rows.append({
            'balls': input_dict['balls'],
            'strikes': input_dict['strikes'],
            'outs_when_up': input_dict['outs'],
            'inning': input_dict['inning'],
            'same_handedness': input_dict['same_handedness'],
            'release_speed': profile['speed'],
            'pfx_x': profile['pfx_x'],
            'pfx_z': profile['pfx_z']
        })
        valid_classes.append(cls)
        
    df_candidates = pd.DataFrame(candidate_rows)
    
    # Get pitch probabilities and run values
    probs = clf.predict_proba(df_candidates)[0]
    expected_rv = reg.predict(df_candidates)
    
    for cls, prob, xrv in zip(valid_classes, probs, expected_rv):
        results.append({
            'pitch_type': cls,
            'selection_probability': round(float(prob), 3),
            'expected_run_value': round(float(xrv), 4)
        })
        
    # Sort results to find mathematically best pitch (lowest expected run value for pitcher)
    sorted_by_optimal = sorted(results, key=lambda x: x['expected_run_value'])
    optimal_pitch = sorted_by_optimal[0]['pitch_type']
    
    return {
        'pitch_evaluations': results,
        'mathematically_optimal_pitch': optimal_pitch
    }
def get_pitch_profiles_from_data(df):
    """
    Groups historical data by pitch type to calculate exact speed 
    and movement averages directly from Statcast.
    """
    profiles = df.groupby('pitch_type')[['release_speed', 'pfx_x', 'pfx_z']].mean().to_dict(orient='index')
    
    # Restructure dictionary format
    return {
        pitch: {
            'speed': metrics['release_speed'],
            'pfx_x': metrics['pfx_x'],
            'pfx_z': metrics['pfx_z']
        }
        for pitch, metrics in profiles.items()
    }