import os
import streamlit as st
import pandas as pd
import streamlit_pandas as sp
from itertools import combinations
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter  # Import FuncFormatter for formatting ticks





st.set_page_config(layout="wide")
@st.cache_data
def load_data():
    file_path = "sales_data.csv"  # Update with the correct path to your CSV file
    df = pd.read_csv(file_path, encoding='latin1')  # Specify the encoding here
    df['ORDER_DATE'] = pd.to_datetime(df['ORDER_DATE'], errors='coerce')  # Convert to datetime
    return df

df = load_data()

# Create month and year columns for grouping
df['MONTH'] = df['ORDER_DATE'].dt.to_period('M')

# Calculate total monthly sales
monthly_sales = df.groupby('MONTH')['SALES'].sum().reset_index()

create_data = {
    "PRICE_EACH": "multiselect",
    "STATUS": "multiselect",
    "PRODUCTLINE": "multiselect",
    "PRODUCTCODE": "multiselect",
    "CUSTOMER_NAME": "text",
    "PHONE": "multiselect",
    "ADDRESSLINE1": "text",
    "CITY": "multiselect",
    "COUNTRY": "multiselect",
    "CONTACTLASTNAME": "text",
    "CONTACTFIRSTNAME": "text",
    "DEALSIZE": "multiselect",
    "POSTALCODE": "multiselect"



}

all_widgets = sp.create_widgets(df, create_data)
res = sp.filter_df(df, all_widgets)


st.header("This is the Data After Sidebar Filtering")
st.write(res)

# Line chart for total monthly sales using st.line_chart

st.header("Important Tables")


res['MonthYear'] = res['ORDER_DATE'].dt.to_period('M')



sales_by_monthyear = res.groupby('MonthYear')['SALES'].sum()
top_10_months = sales_by_monthyear.nlargest(10)
# Display the top 10 months and their sales using Streamlit



sales_by_CUSTOMER_NAME = res.groupby('CUSTOMER_NAME')['SALES'].sum()
sales_by_CUSTOMER_NAME_10_months = sales_by_CUSTOMER_NAME.nlargest(10)
# Display the top 10 months and their sales using Streamlit


# Group by 'CITY' and sum 'SALES'
sales_by_city = res.groupby('CITY')['SALES'].sum()
top_10_cities = sales_by_city.nlargest(10)

col1, col2, col3 = st.columns(3)

with col1:
    st.write("Top 10 Months by Total Sales:")
    st.write(top_10_months)

with col2:
    st.write("Top 10 Customers by Total Sales:")
    st.write(sales_by_CUSTOMER_NAME_10_months)

with col3:
    st.write("Top 10 Cities by Total Sales:")
    st.write(top_10_cities)



sales_by_delay = res.groupby('STATUS')['SALES'].count()
sales_by_delaytop5 = sales_by_delay.nlargest(10)


sales_by_postal = res.groupby('POSTALCODE')['SALES'].sum()
sales_by_postaltop = sales_by_postal.nlargest(10)


sales_by_prodd = res.groupby('PRODUCTLINE')['SALES'].sum()
sales_by_prodtop10 = sales_by_prodd.nlargest(10)




def get_top_ten_common_pairs(df):
    pairs_counter = Counter()
    
    # Group by ORDER_NUMBER and find pairs
    grouped = df.groupby('ORDER_NUMBER')['PRODUCTLINE'].apply(list)
    
    for products in grouped:
        pairs = combinations(products, 2)
        pairs_counter.update(pairs)
    
    # Get the top ten most common pairs
    top_ten_pairs = pairs_counter.most_common(20)
    return top_ten_pairs

# Get the top ten most common pairs
top_ten_pairs = get_top_ten_common_pairs(res)

# Create a DataFrame for the top ten pairs
comdf = pd.DataFrame(top_ten_pairs, columns=['Most common pair', 'Counts'])
comdf['Most common pair'] = comdf['Most common pair'].apply(lambda x: f"{x[0]} & {x[1]}")







col1, col2, col3 , col4 = st.columns(4)

with col1:

    st.write("Transportation Progress:")
    st.write(sales_by_delaytop5)

with col2:
    st.write("Top 10 Zip Codes by Total Sales:")
    st.write(sales_by_postaltop)

with col3:
    st.write("Top 10 Products by Total Sales:")
    st.write(sales_by_prodtop10)

with col4:
    st.write("Most common pairs of items bought:")
    st.dataframe(comdf)



avgysales = res.groupby(by="YEAR")['SALES'].sum().reset_index()
avgmsales = res.groupby(by="MONTH")['SALES'].sum().reset_index()

# Streamlit app
st.title("Sales Analysis")

# Create two columns
col1, col2 = st.columns(2)

# Column 1: Average Yearly Sales
with col1:
    st.header("Average Yearly Sales")
    st.line_chart(data=avgysales, x='YEAR', y='SALES')

# Column 2: Average Monthly Sales
with col2:
    st.header("Average Monthly Sales")
    st.line_chart(data=avgmsales, x='MONTH', y='SALES')

avgsales = df.groupby(by="YEAR")['SALES'].sum()
stdsales = df.groupby(by="YEAR")['SALES'].std()

# Get the three most recent years and their average sales and standard deviations
recent_years = avgsales.index[-3:]
recent_avg_sales = avgsales[-3:]
recent_std_sales = stdsales[-3:]

# Optionally, you can format this in a more structured layout
col1, col2, col3 = st.columns(3)
col1.metric(label=f"Year {recent_years[0]}", 
            value=f"${recent_avg_sales[recent_years[0]]:.2f}", 
            delta=f"Std Dev: ${recent_std_sales[recent_years[0]]:.1f}")
col2.metric(label=f"Year {recent_years[1]}", 
            value=f"${recent_avg_sales[recent_years[1]]:.2f}", 
            delta=f"Std Dev: ${recent_std_sales[recent_years[1]]:.1f}")
col3.metric(label=f"Year {recent_years[2]}", 
            value=f"${recent_avg_sales[recent_years[2]]:.2f}", 
            delta=f"Std Dev: ${recent_std_sales[recent_years[2]]:.1f}")









st.header("Total Monthly Sales - Line Chart")


# Plotting with Matplotlib
plt.figure(figsize=(12, 6))
plt.plot(sales_by_monthyear.index.astype(str), sales_by_monthyear.values, marker='o', linestyle='-')
plt.title('Total Sales Over Time (Monthly)')
plt.xlabel('Month-Year')
plt.ylabel('Total Sales')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.gca().get_yaxis().set_major_formatter(FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))
st.pyplot(plt)
