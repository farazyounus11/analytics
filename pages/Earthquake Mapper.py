import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk



st.title("Earthquake Visualization")
st.header('By Faraz Younus | M.S. Stats & Data Science', divider='gray')

st.write("You can use the sidebar to select Severity")


import streamlit as st
import pandas as pd

# Assuming 'date' column is in a format like 'YYYY-MM-DD' or similar that pandas can parse
df = pd.read_csv("earthq.csv")
df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True, errors='coerce')

# Load your dataset
min_depth, max_depth = df['depth'].min(), df['depth'].max()
min_mag, max_mag = df['mag'].min(), df['mag'].max()

# Sidebar sliders for Date, Depth, and Magnitude
min_date, max_date = df['date'].min().to_pydatetime(), df['date'].max().to_pydatetime()

# Use these Python datetime objects with the Streamlit slider
selected_date_range = st.sidebar.slider(
    "Select Date Range",
    value=(min_date, max_date),
    format="MM/DD/YYYY"  # or any format you prefer
)

selected_depth_range = st.sidebar.slider(
    "Select Depth Range",
    min_value=float(min_depth),
    max_value=float(max_depth),
    value=(min_depth, max_depth)
)

selected_mag_range = st.sidebar.slider(
    "Select Earthquake Magnitude Range",
    min_value=float(min_mag),
    max_value=float(max_mag),
    value=(min_mag, max_mag)
)

# Filter the DataFrame based on the selected range of dates, depths, and magnitudes
filtered_df = df[
    (df['date'] >= selected_date_range[0]) & 
    (df['date'] <= selected_date_range[1]) &
    (df['depth'] >= selected_depth_range[0]) & 
    (df['depth'] <= selected_depth_range[1]) &
    (df['mag'] >= selected_mag_range[0]) & 
    (df['mag'] <= selected_mag_range[1])
]

st.write(filtered_df)



st.header('Map', divider='gray')
st.pydeck_chart(pdk.Deck(
    map_style=None,
    initial_view_state=pdk.ViewState(
        latitude=40.10059,
        longitude=-82.925194,
        zoom=2,
        pitch=50,
    ),
    layers=[
        pdk.Layer(
            'HexagonLayer',
            data=filtered_df,
            get_position='[lon, lat]',
            radius=17000,
            elevation_scale=3,
            elevation_range=[0, 1000],
            get_color='[200, 30, 0, 160]',
            pickable=True,
            extruded=True,
        ),
        pdk.Layer(
            'ScatterplotLayer',
            data=filtered_df,
            get_position='[lon, lat]',
            get_color='[200, 30, 0, 160]',
            get_radius=17000,
        ),
    ],
))