import streamlit as st
import pandas as pd
import streamlit_pandas as sp
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os 
st.set_page_config(layout="wide")

current_directory = os.path.dirname(__file__)
os.chdir(current_directory)

@st.cache
def load_data():
    df = pd.read_csv("gdpdata.csv")
    return df

df1 = load_data()
df1.columns = df1.columns.str.capitalize()

selected_countries = st.multiselect("Select countries", df1["Country"].unique())
selected_continents = st.multiselect("Select continents", df1["Continent"].unique())

filtered_df = df1[df1["Country"].isin(selected_countries) & df1["Continent"].isin(selected_continents)]

st.markdown("## Use the Side-bar to select countries to View")
col1, col2 = st.columns(2)
with col1:
    st.subheader('Life Expectancy')
    st.line_chart(filtered_df, x="Year", y="Life_exp", color="Country")

    st.subheader('HDI Index')
    st.line_chart(filtered_df, x="Year", y="Hdi_index", color="Country")

with col2:
    st.subheader('CO2 Consumption')
    st.line_chart(filtered_df, x="Year", y="Co2_consump", color="Country")

    st.subheader('GDP')
    st.line_chart(filtered_df, x="Year", y="Gdp", color="Country")



def run_predictions():
    X = df['Year'].values.reshape(-1, 1)  # Features (Year)
    columns_to_predict = ['Life_exp', 'Hdi_index', 'Co2_consump', 'Gdp', 'Services']
    predictions = {}

    for column in columns_to_predict:
        y = df[column].values  # Target variable
        model = LinearRegression()
        model.fit(X, y)
        future_years = np.arange(2019, 2029).reshape(-1, 1)
        predictions[column] = model.predict(future_years)

    return predictions

# Display predictions using Streamlit
st.title('Run Predictions for 2019-2029')

# Add a button to trigger predictions
if st.button('Run Predictions'):
    predictions = run_predictions()

    # Split the predictions into two groups
    predictions_group1 = {column: prediction for i, (column, prediction) in enumerate(predictions.items()) if i % 2 == 0}
    predictions_group2 = {column: prediction for i, (column, prediction) in enumerate(predictions.items()) if i % 2 != 0}

    col1, col2 = st.columns(2)

    with col1:
        for column, prediction in predictions_group1.items():
            st.subheader(column)
            st.line_chart(pd.DataFrame({column: prediction}, index=np.arange(2019, 2029)))

    with col2:
        for column, prediction in predictions_group2.items():
            st.subheader(column)
            st.line_chart(pd.DataFrame({column: prediction}, index=np.arange(2019, 2029)))
