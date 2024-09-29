import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np

# Set the title of the app
st.title("Data Visualization Portfolio")

# Load sample data
df = pd.DataFrame({
    'Category': ['A', 'B', 'C', 'D'],
    'Values': [4, 3, 2, 5]
})

# Create a sidebar for navigation
st.sidebar.title("Visualization Selection")
visualization_type = st.sidebar.selectbox(
    "Choose a visualization type:",
    ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot"]
)

# Function to display the selected visualization
def show_visualization(vis_type):
    if vis_type == "Bar Chart":
        st.subheader("Bar Chart")
        plt.bar(df['Category'], df['Values'], color='skyblue')
        st.pyplot(plt)
    
    elif vis_type == "Line Chart":
        st.subheader("Line Chart")
        plt.plot(df['Category'], df['Values'], marker='o')
        st.pyplot(plt)

    elif vis_type == "Scatter Plot":
        st.subheader("Scatter Plot")
        x = np.random.rand(50)
        y = np.random.rand(50)
        plt.scatter(x, y, alpha=0.5)
        st.pyplot(plt)

    elif vis_type == "Histogram":
        st.subheader("Histogram")
        data = np.random.randn(1000)
        plt.hist(data, bins=30, color='purple', alpha=0.7)
        st.pyplot(plt)

    elif vis_type == "Box Plot":
        st.subheader("Box Plot")
        sns.boxplot(x='Category', y='Values', data=df)
        st.pyplot(plt)

# Show the selected visualization
show_visualization(visualization_type)

# Add an explanation or note section
st.sidebar.subheader("About this Portfolio")
st.sidebar.text("This app showcases various types of visualizations created with Streamlit. "
                "You can select different visualizations from the sidebar.")
