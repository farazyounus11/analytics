
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans  # Example clustering algorithm
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import numpy as np


import os

target_directory = '/mount/src/analytics/pages/'
os.chdir(target_directory)
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df['price'] = df[['open', 'high', 'low', 'close']].mean(axis=1)
    df['variance'] = df[['open', 'high', 'low', 'close']].var(axis=1)
    df['market_cap'] = df['price'] * df['volume']
    return df
st.set_page_config(layout="wide")

# Path to your CSV file
file_path = "s&p500.csv"
df = load_data(file_path)


st.title('PCA Clustering of Stocks')
valid_columns = [col for col in df.columns if col not in ['open','close', 'high', 'low', 'Name', 'date']]

value_column = st.selectbox("Select a Metric for Correlation:", valid_columns )

pivot_table = df.pivot(index='date', columns='Name', values=value_column)

# Filter columns with less than or equal to 400 NaNs
na_counts = pivot_table.isna().sum()
columns_to_keep = na_counts[na_counts <= 400].index
pivot_table = pivot_table[columns_to_keep]


pivot_table = pivot_table.fillna(pivot_table.median())
scaler = StandardScaler()
pivot_table_scaled = scaler.fit_transform(pivot_table)

scaled_pivot_table = pivot_table_scaled.corr()

##PCA
pca = PCA(n_components=2)  # Choose number of principal components
principal_components = pca.fit_transform(scaled_pivot_table.T)
principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'], index=scaled_pivot_table.columns)

# Perform K-means clustering
kmeans = KMeans(n_clusters=10)  # Specify number of clusters
clusters = kmeans.fit_predict(principal_df)
principal_df['Cluster'] = clusters

# Get stock names and their corresponding clusters
stock_clusters = principal_df[['Cluster']]
principal_df['Name'] = principal_df.index

pivot_table_corr = scaled_pivot_table.corr()




fig = px.scatter(principal_df, x='PC1', y='PC2', color='Cluster', text='Name',
		 labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
		 width=1200, height=800, color_continuous_scale='Viridis')  # Set width and height

# Update the layout to adjust the text position
fig.update_traces(textposition='top center')

# Display the Plotly figure in Streamlit
st.plotly_chart(fig)

st.title('Correlation Analysis')
st.markdown("### Analysis after scaling the price data to be between -2 and 3. Scaling is done so that stocks priced in the thousends are comparable to stocks priced in tens or ones")

stock = st.selectbox('Select a stock:', pivot_table_corr.columns )

# Calculate negative correlations
stock_correlations_neg = pivot_table_corr[stock].sort_values()
top_5_negatively_correlated = stock_correlations_neg.head(10)

# Calculate positive correlations
stock_correlations_pos = pivot_table_corr[stock].sort_values(ascending=False)
top_10_positively_correlated = stock_correlations_pos.head(10)

# Display the results using st.columns


st.write(f"#### Top 10 most negatively correlated stocks to {stock}:")
col1, col2 = st.columns([1, 3])

with col1:
    st.write(top_5_negatively_correlated)

with col2:
	top_5_names = top_5_negatively_correlated.index
	st.line_chart(scaled_pivot_table[top_5_names])

st.write(f"#### Top 10 most positively correlated stocks to {stock}:")
col11, col22 = st.columns([1, 3])

with col11:
    st.write(top_10_positively_correlated)

with col22:
	top_5_names = top_10_positively_correlated.index
	st.line_chart(scaled_pivot_table[top_5_names])




