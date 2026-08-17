import duckdb
import streamlit as st

st.title("⚾ Statcast Data Explorer")

# Interactive Filters
min_speed = st.slider("Min Release Speed (mph)", 80, 105, 90)
selected_pitch = st.selectbox(
    "Pitch Type", ["FF", "SL", "CH", "CU", "SI", "FC"]
)

# Run Query Dynamic to Filters
query = f"""
SELECT 
    m.name AS pitcher_name,
    s.pitch_type,
    s.release_speed,
    s.balls,
    s.strikes,
    s.pfx_x,
    s.pfx_z
FROM 'data/statcast_2025.parquet' s
JOIN 'data/player_map.parquet' m ON s.pitcher = m.player_id
WHERE s.release_speed >= {min_speed}
  AND s.pitch_type = '{selected_pitch}'
LIMIT 100
"""

df = duckdb.query(query).to_df()

# Render Interactive Grid
st.dataframe(df, use_container_width=True)