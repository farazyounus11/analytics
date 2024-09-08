import streamlit as st
import pandas as pd
import pydeck as pdk


@st.cache_data
def load_comdf3():
    df = pd.read_csv("ecomm1.csv")
    df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])    
    return df
    
comdf = load_comdf3()

product_categories = comdf['Product Category'].unique()
col1, col2, col3 = st.columns(3)

with col1:
    selected_categories = st.multiselect(
        "Select Product Categories",
        options=product_categories,
        default=['Automotive Parts']
    )

with col2:
    purchase_status = st.multiselect(
        "Purchase Status",
        options=comdf['Purchase Completed'].unique(),
        default=['Completed']
    )




if selected_categories:
    com_filtered_df = comdf[comdf['Product Category'].isin(selected_categories)]

if purchase_status:
    com_filtered_df = com_filtered_df[com_filtered_df['Purchase Completed'].isin(purchase_status)]

with col3:
    st.metric(label="Number of Transactions", value=len(com_filtered_df))


if not com_filtered_df.empty:
    st.metric(label="Number of Transactions", value=len(com_filtered_df))
    transaction_counts_by_date = com_filtered_df.groupby(['Transaction Date', 'Product Category']).size().unstack(fill_value=0)
    st.header('Map', divider='gray')
    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=com_filtered_df['Latitude'].mean(),
            longitude=com_filtered_df['Longitude'].mean(),
            zoom=11,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
                'HexagonLayer',
                data=com_filtered_df,
                get_position='[Longitude, Latitude]',
                radius=100,
                elevation_scale=4,
                elevation_range=[0, 1000],
                pickable=True,
                extruded=True,
            ),
            pdk.Layer(
                'ScatterplotLayer',
                data=com_filtered_df,
                get_position='[Longitude, Latitude]',
                get_color=[200, 30, 0, 160],
                get_radius=100,
            ),
        ],
    ))
    st.line_chart(transaction_counts_by_date)
else:
    st.warning("No data available for selected filters")
