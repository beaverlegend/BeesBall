import os
import pandas as pd
import numpy as np
from pybaseball import statcast

# Ensure target directories exist
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

RAW_PARQUET_PATH = os.path.join(DATA_DIR, "statcast_2025.parquet")
PROFILES_PARQUET_PATH = os.path.join(DATA_DIR, "pitcher_profiles.parquet")
PLAYER_MAP_PATH = os.path.join(DATA_DIR, "player_map.parquet")


def fetch_and_clean_data(start_date="2025-04-01", end_date="2025-10-01"):
    """
    1. Fetches raw Statcast data via pybaseball.
    2. Cleans missing values, converts types, and engineers features.
    3. Saves primary training data to data/statcast_2025.parquet.
    """
    print(f"Fetching Statcast pitch data between {start_date} and {end_date}...")
    df = statcast(start_dt=start_date, end_dt=end_date)
    
    if df.empty:
        raise ValueError("No data returned from Statcast for the specified date range.")

    print(f"Raw rows downloaded: {len(df)}")

    # 1. Drop rows missing critical labels
    df = df.dropna(subset=['pitch_type', 'balls', 'strikes', 'p_throws', 'stand', 'type'])
    
    # 2. Convert game state fields to integers
    df['balls'] = df['balls'].astype(int)
    df['strikes'] = df['strikes'].astype(int)
    df['outs_when_up'] = df['outs_when_up'].astype(int)
    df['inning'] = df['inning'].astype(int)

    # 3. Engineer Feature: Handedness / Platoon Advantage
    # 1 if pitcher and batter share handedness (R/R or L/L), 0 if opposite (R/L or L/R)
    df['same_handedness'] = (df['p_throws'] == df['stand']).astype(int)

    # 4. Fill numeric pitch characteristics missing values
    df['release_speed'] = df['release_speed'].fillna(df['release_speed'].median())
    df['pfx_x'] = df['pfx_x'].fillna(0.0)
    df['pfx_z'] = df['pfx_z'].fillna(0.0)
    df['delta_run_exp'] = df['delta_run_exp'].fillna(0.0)

    # Save cleaned primary training set
    df.to_parquet(RAW_PARQUET_PATH, index=False)
    print(f"Cleaned dataset saved to: {RAW_PARQUET_PATH}")
    
    return df


def generate_pitcher_profiles(df):
    """
    Groups data by pitcher ID and pitch type to calculate exact speed 
    and movement profiles for every pitcher in the database.
    """
    print("Generating pitcher velocity and movement profiles...")
    
    # Group by Pitcher ID and Pitch Code
    profiles = df.groupby(['pitcher', 'pitch_type']).agg(
        avg_speed=('release_speed', 'mean'),
        avg_pfx_x=('pfx_x', 'mean'),
        avg_pfx_z=('pfx_z', 'mean'),
        pitch_count=('pitch_type', 'count')
    ).reset_index()

    # Calculate overall usage percentage per pitch type for each pitcher
    pitcher_totals = profiles.groupby('pitcher')['pitch_count'].transform('sum')
    profiles['usage_rate'] = (profiles['pitch_count'] / pitcher_totals).round(3)

    # Round metrics for fast serialization
    profiles['avg_speed'] = profiles['avg_speed'].round(2)
    profiles['avg_pfx_x'] = profiles['avg_pfx_x'].round(2)
    profiles['avg_pfx_z'] = profiles['avg_pfx_z'].round(2)

    profiles.to_parquet(PROFILES_PARQUET_PATH, index=False)
    print(f"Pitcher profiles saved to: {PROFILES_PARQUET_PATH}")


def generate_player_map(df):
    """
    Creates a lookup table mapping MLBAM Player IDs to Player Names.
    """
    print("Generating player ID lookup mapping...")
    
    # Extract unique pitcher mappings
    pitchers = df[['pitcher', 'player_name']].drop_duplicates().rename(
        columns={'pitcher': 'player_id', 'player_name': 'name'}
    )
    
    # Extract unique batter mappings (if player_name represents pitcher, batter is ID)
    batters = df[['batter']].drop_duplicates().rename(columns={'batter': 'player_id'})
    batters['name'] = "Unknown Batter " + batters['player_id'].astype(str)
    
    player_map = pd.concat([pitchers, batters]).drop_duplicates(subset=['player_id'])
    player_map.to_parquet(PLAYER_MAP_PATH, index=False)
    print(f"Player map saved to: {PLAYER_MAP_PATH}")


def run_pipeline(start_date="2025-04-01", end_date="2025-10-01"):
    """
    Master runner script to refresh all database tables in order.
    """
    df = fetch_and_clean_data(start_date=start_date, end_date=end_date)
    generate_pitcher_profiles(df)
    generate_player_map(df)
    print("Data processing complete.")


if __name__ == "__main__":
    run_pipeline()