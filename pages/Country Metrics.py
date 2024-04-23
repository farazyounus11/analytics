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

@st.cache_data
def load_data():
    df = pd.read_csv("gdpdata.csv")
    return df

df1 = load_data()
df1.columns = df1.columns.str.capitalize()


print(df1.columns)
create_data = {
                "Country": "multiselect",
                "Continent": "multiselect"}

all_widgets = sp.create_widgets(df1, create_data)


df = sp.filter_df(df1, all_widgets)


st.markdown("## Use the Side-bar to select countries to View")
import streamlit as st

# Assuming df is your DataFrame containing the data
col1, col2 = st.columns(2)
with col1:
    st.subheader('Life Expectancy')
    st.line_chart(df, x="Year", y="Life_exp", color="Country")
with col1:
    st.subheader('HDI Index')
    st.line_chart(df, x="Year", y="Hdi_index", color="Country")

with col2:
    st.subheader('CO2 Consumption')
    st.line_chart(df, x="Year", y="Co2_consump", color="Country")
with col2:
    st.subheader('GDP')
    st.line_chart(df, x="Year", y="Gdp", color="Country")

st.markdown("## Below Select a Metirc for Heatmap")


selected_variable = st.selectbox('Select Variable', ['Gdp', 'Co2_consump', 'Hdi_index', 'Life_exp'])

df_2018 = df[df['Year'] == 2018]

# Calculate the maximum value for the selected variable by country for the year 2018
max_values_by_country = df_2018.groupby('Country')[selected_variable].max().reset_index()

# Define the color scale for each variable
color_scales = {
    'Gdp': 'Viridis',
    'Co2_consump': 'YlOrRd',
    'Hdi_index': 'Plasma',
    'Life_exp': 'YlOrRd'
}

# Create the choropleth map using the maximum value for each country in 2018
fig = px.choropleth(max_values_by_country, 
                    locations='Country',  
                    locationmode='country names',  
                    color=selected_variable,  # Dynamically change the color variable
                    hover_name='Country',
                    projection='natural earth',  
                    title=f'Maximum {selected_variable.replace("_", " ").title()} in 2018 by Country',
                    color_continuous_scale=color_scales[selected_variable])

st.plotly_chart(fig)






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