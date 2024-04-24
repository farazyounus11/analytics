import streamlit as st
import pandas as pd
import streamlit_pandas as sp
import plotly.express as px

st.markdown("# Country Mapper Created by Faraz")
st.markdown("## Use the side bar to filter and use the dropdown menu to contrast")



@st.cache_data
def load_data():
    df1 = pd.read_csv("worlddata.csv", dtype=str)

    # Replace all non-alphanumeric characters with an empty string
    df1 = df1.replace(r'[^\w.]', '', regex=True)
    # Convert columns except the first one to floats
    for col in df1.columns[1:]:
        df1[col] = pd.to_numeric(df1[col], errors='coerce')

    # Fill NaN values with column mean
    df1 = df1.fillna(df1.mean())

    # Round floats to one decimal place
    df1 = df1.round(1)

    return df1

df1 = load_data()

# Capitalize column names
df1.columns = df1.columns.str.capitalize()

# Create widgets
create_data = {"Country": "multiselect"}

all_widgets = sp.create_widgets(df1, create_data, ignore_columns=["Latitude", "Longitude"])

# Filter DataFrame
df = sp.filter_df(df1, all_widgets)

variable_names = df.columns[1:]
selected_variable = st.selectbox('Select Variable', variable_names)

default_color_scale = 'Viridis'
color_scales = {variable: default_color_scale for variable in variable_names}
color_scale = color_scales.get(selected_variable, default_color_scale)

# Use the filtered DataFrame `df` for plotting
fig = px.choropleth(df, 
                    locations='Country',  
                    locationmode='country names',  
                    color=selected_variable,
                    hover_name='Country',
                    projection='natural earth',  
                    title=f'{selected_variable.replace("_", " ").title()} by Country',
                    color_continuous_scale=color_scale,
                    width=1000,  # Adjust width as needed
                    height=600   # Adjust height as needed
)
st.plotly_chart(fig)


st.header('Use the Drop-Down Menu to Select the Countries for Line Chart', divider='red')

@st.cache
def load_data():
    df = pd.read_csv("gdpdata.csv")
    return df

original_df = load_data()
original_df.columns = original_df.columns.str.capitalize()

selected_countries = st.multiselect("Select countries", original_df["Country"].unique())

filtered_df = original_df[original_df["Country"].isin(selected_countries)]

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
