import streamlit as st
import pandas as pd
import pydeck as pdk

comdf = pd.read_csv("ecomm1.csv")
comdf['Transaction Date'] = pd.to_datetime(comdf['Transaction Date'])

product_categories = comdf['Product Category'].unique()
col1, col2 = st.columns(2)

# In the first column, add the product categories multiselect
with col1:
    selected_categories = st.multiselect(
        "Select Product Categories", 
        options=product_categories, 
        default=['Automotive Parts'])
    
with col2:
    purchase_status = st.multiselect(
        "Purchase Status", 
        options=comdf['Purchase Completed'].unique(), 
        default=['Completed'])

if selected_categories:
    filtered_df = comdf[comdf['Product Category'].isin(selected_categories)]

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
    st.write(transaction_counts_by_date)
else:
    st.warning("No data available for selected filters")
