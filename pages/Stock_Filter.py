import streamlit as st
import pandas as pd
import streamlit_pandas as sp
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


st.set_page_config(layout="wide")
@st.cache_data
def load_data():
    df = pd.read_csv(file)
    return df


st.header("Sidebar lets users Filter for Certain Financial Requirements in a Company")

file = "stock.csv"
df = load_data()

df = df.fillna(df.median())



create_data = {

                "Industry": "multiselect",
                "Sector": "multiselect",
                "Name": "multiselect",
                "RecommendationKey": "multiselect"}


all_widgets = sp.create_widgets(df, create_data, ignore_columns=["Ticker", "Name","RecommendationMean" , "City", "State"])
res = sp.filter_df(df, all_widgets)



st.markdown("### The percentage of companies meet your conditions:")

percentage = round((res.shape[0] / df.shape[0]) * 100, 1)

# Use Streamlit to display the metric
st.metric(label="Percentage", value=f"{percentage}%")


st.write(res['Name'])


numerical_columns = df.select_dtypes(include=['number']).columns


# Streamlit app
st.title("Where Does Your Company Rank")

# Select a row based on the Name column
selected_name = st.selectbox("Select a Company", df['Name'])

# Extract the row corresponding to the selected company
selected_row = df[df['Name'] == selected_name].squeeze()

# Create two columns
col1, col2 = st.columns(2)

# Dropdown menu for selecting numerical columns in each column
selected_column1 = col1.selectbox("Select a Metric for Ranking", numerical_columns, key='col1')
selected_column2 = col2.selectbox("Select a Metric for Ranking", numerical_columns, key='col2')

# Plot histogram in the first column
if selected_column1:
    col1.write(f"Histogram of {selected_column1}")
    fig1, ax1 = plt.subplots()
    ax1.hist(df[selected_column1], bins=30, edgecolor='black')
    company_value1 = selected_row[selected_column1]
    ax1.axvline(company_value1, color='red', linestyle='dashed', linewidth=2, label=f'{selected_name} value: {company_value1}')
    ax1.set_xlabel(selected_column1)
    ax1.set_ylabel('Frequency')
    ax1.legend()
    col1.pyplot(fig1)

# Plot histogram in the second column
if selected_column2:
    col2.write(f"Histogram of {selected_column2}")
    fig2, ax2 = plt.subplots()
    ax2.hist(df[selected_column2], bins=30, edgecolor='black')
    company_value2 = selected_row[selected_column2]
    ax2.axvline(company_value2, color='red', linestyle='dashed', linewidth=2, label=f'{selected_name} value: {company_value2}')
    ax2.set_xlabel(selected_column2)
    ax2.set_ylabel('Frequency')
    ax2.legend()
    col2.pyplot(fig2)


st.title('Top Companies Analysis')

# Layout in columns
col1, col2, col3 = st.columns(3)

# Column 1: Top 20 Companies by Net Income to Common Shareholders
with col1:
    st.markdown('### Top 20 Companies by Net Income to Common Shareholders')
    top_20_net_income = get_top_20_names(df, 'NetIncomeToCommon')
    st.write(top_20_net_income)

# Column 2: Top 20 Companies by Earnings Quarterly Growth
with col2:
    st.markdown('### Top 20 Companies by Earnings Quarterly Growth')
    top_20_earnings_growth = get_top_20_names(df, 'EarningsQuarterlyGrowth')
    st.write(top_20_earnings_growth)

# Column 3: Top 20 Companies by Forward PE
with col3:
    st.markdown('### Top 20 Companies by Forward PE')
    top_20_forward_pe = get_top_20_names(df, 'ForwardEps')
    st.write(top_20_forward_pe)
