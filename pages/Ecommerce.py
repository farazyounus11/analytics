import streamlit as st
import pandas as pd
import pydeck as pdk

st.set_page_config(layout="wide")

# Load the ecomm CSV file
@st.cache_data
def load_dataframe(file_path):
    df = pd.read_csv(file_path)
    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])
    return df

# Path to your CSV file
file_path = "ecomm1.csv"  # Replace with your actual path if different
df = load_dataframe(file_path)

st.markdown("## E-commerce Transactions Visualization")
st.markdown("### Use the Sidebar to Filter Data")

try:
    min_date = df['Transaction Date'].min().to_pydatetime()
    max_date = df['Transaction Date'].max().to_pydatetime()
    selected_start_date, selected_end_date = st.sidebar.slider(
        "Select date range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date)
    )
except Exception as e:
    st.error(f"Error with date slider: {str(e)}")

# Filter by Product Category
product_categories = df['Product Category'].unique()
selected_categories = st.sidebar.multiselect("Select Product Categories", options=product_categories, default=['Automotive Parts'])

# Filter by Purchase Completed
purchase_status = st.sidebar.multiselect("Purchase Status", options=df['Purchase Completed'].unique(), default=['Completed'])

# Apply filters
filtered_df = df[
    (df['Transaction Date'] >= selected_start_date) & 
    (df['Transaction Date'] <= selected_end_date)
]

if selected_categories:
    filtered_df = filtered_df[filtered_df['Product Category'].isin(selected_categories)]

if purchase_status:
    filtered_df = filtered_df[filtered_df['Purchase Completed'].isin(purchase_status)]

if not filtered_df.empty:
    st.metric(label="Number of Transactions", value=len(filtered_df))
    transaction_counts_by_date = filtered_df.groupby(['Transaction Date', 'Product Category']).size().unstack(fill_value=0)
    st.header('Map', divider='gray')
    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=filtered_df['Latitude'].mean(),
            longitude=filtered_df['Longitude'].mean(),
            zoom=11,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
                'HexagonLayer',
                data=filtered_df,
                get_position='[Longitude, Latitude]',
                radius=100,
                elevation_scale=4,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
            ),
            pdk.Layer(
                'ScatterplotLayer',
                data=filtered_df,
                get_position='[Longitude, Latitude]',
                get_color=[200, 30, 0, 160],
                get_radius=100,
            ),
        ],
    ))
    st.line_chart(transaction_counts_by_date)
else:
    st.warning("No data available for selected filters")
