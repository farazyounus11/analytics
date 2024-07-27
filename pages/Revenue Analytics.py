import os
import streamlit as st
import pandas as pd
import streamlit_pandas as sp
from itertools import combinations
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter  # Import FuncFormatter for formatting ticks
import plotly.express as px
import altair as alt


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

column_to_move = df.pop("PRODUCTLINE")
df.insert(0, "PRODUCTLINE", column_to_move)

column_to_move = df.pop("CUSTOMER_NAME")
df.insert(0, "CUSTOMER_NAME", column_to_move)

create_data = {
    "ORDER_NUMBER": "multiselect",
    "PRICE_EACH": "multiselect",
    "STATUS": "multiselect",
    "PRODUCTLINE": "multiselect",
    "PRODUCTCODE": "multiselect",
    "CUSTOMER_NAME": "multiselect",
    "PHONE": "multiselect",
    "ADDRESSLINE1": "text",
    "CITY": "multiselect",
    "COUNTRY": "multiselect",
    "CONTACTLASTNAME": "text",
    "CONTACTFIRSTNAME": "text",
    "DEALSIZE": "multiselect",
    "POSTALCODE": "multiselect"}

all_widgets = sp.create_widgets(df, create_data, ignore_columns=["CONTACTLASTNAME", "CONTACTFIRSTNAME"])

try:
    res = sp.filter_df(df, all_widgets)

    
except KeyError as e:
    # Handle the KeyError, e.g., column does not exist in DataFrame
    st.error(f"An error occurred: {e}. Please check your input and try again.")
    st.stop()  # Stop further execution of the script

except Exception as e:
    # Handle any other unexpected exceptions
    st.error(f"An unexpected error occurred: {e}. Please try again later.")
    st.stop()  # Stop further execution of the script




st.markdown("### In this interactive e-commerce analytics app, you can use the sidebar to filter for Customer_Name, Order_number, Order_Number etc. The app automatically updates")
st.markdown("### If the app produces an error, that's because there at too many filters active")



avgsales = res.groupby(by="YEAR")['SALES'].sum()
stdsales = res.groupby(by="YEAR")['SALES'].std()

# Get the three most recent years and their average sales and standard deviations
recent_years = avgsales.index[-3:]
recent_avg_sales = avgsales[-3:]
recent_std_sales = stdsales[-3:]

col1, col2, col3 = st.columns(3)

# Check the length of recent_years and display metrics accordingly
if len(recent_years) > 0:
    col1.metric(label=f"Year {recent_years[0]}", 
                value=f"${round(recent_avg_sales[recent_years[0]])}", 
                delta=f"Std Dev: ${round(recent_std_sales[recent_years[0]])}")
if len(recent_years) > 1:
    col2.metric(label=f"Year {recent_years[1]}", 
                value=f"${round(recent_avg_sales[recent_years[1]])}", 
                delta=f"Std Dev: ${round(recent_std_sales[recent_years[1]])}")
if len(recent_years) > 2:
    col3.metric(label=f"Year {recent_years[2]}", 
                value=f"${round(recent_avg_sales[recent_years[2]])}", 
                delta=f"Std Dev: ${round(recent_std_sales[recent_years[2]])}")
else:
    col3.metric(label="Year N/A", value="N/A", delta="N/A")

st.metric(label="Numer of Transactions", value=res.shape[0])


if st.checkbox("Show filtered table"):
    st.dataframe(res)







# Line chart for total monthly sales using st.line_chart
st.header("Important Tables")
res['MonthYear'] = res['ORDER_DATE'].dt.to_period('M')

sales_by_monthyear = res.groupby('MonthYear')['SALES'].sum()
top_10_months = sales_by_monthyear.nlargest(10)
# Display the top 10 months and their sales using Streamlit

sales_by_CUSTOMER_NAME = df.groupby('CUSTOMER_NAME')['SALES'].sum()
sales_by_CUSTOMER_NAME_10_months = sales_by_CUSTOMER_NAME.nlargest(10).reset_index()



sales_by_city = res.groupby('CITY')['SALES'].sum()
top_10_cities = sales_by_city.nlargest(10)


col1, col2, col3 = st.columns([1,2,1])
with col1:
    st.write("Top 10 Months by Total Sales:")
    st.write(top_10_months)
with col2:
    st.write("Top 10 Customers by Total Sales:")
    st.bar_chart(sales_by_CUSTOMER_NAME_10_months.set_index('CUSTOMER_NAME'), horizontal=True)
with col3:
    st.write("Top 10 Cities by Total Sales:")
    st.write(top_10_cities)




sales_by_delay = res.groupby('STATUS')['SALES'].count()
sales_by_delaytop5 = sales_by_delay.nlargest(10)





sales_by_postal = res.groupby('POSTALCODE')['SALES'].sum()
sales_by_postaltop = sales_by_postal.nlargest(10)




countryresult = res.groupby('COUNTRY')['SALES'].sum().reset_index()
color_scale = px.colors.sequential.Viridis
fig = px.choropleth(countryresult, 
                    locations='COUNTRY',  
                    locationmode='country names',  
                    color="SALES",
                    hover_name='COUNTRY',
                    title='Interactive Map of Sales by Country',  # Corrected title
                    color_continuous_scale=color_scale,
                    width=600,  # Adjust width as needed
                    height=400 )

fig.update_layout(
    geo=dict(
        scope='world',  # Focus on the world map
        projection_scale=2,  # Adjust projection scale for zoom (1 is default, >1 zooms in, <1 zooms out)
        center=dict(lat=28.675, lon=-40.70)  # Center of the map (lat and lon can be adjusted as needed)
    )
)
# Assuming `st` refers to Streamlit

st.header("Total Monthly Sales")
plt.figure(figsize=(12, 4))
plt.plot(sales_by_monthyear.index.astype(str), sales_by_monthyear.values, marker='o', linestyle='-')
plt.title('Total Sales Over Time (Monthly)')
plt.xlabel('Month-Year')
plt.ylabel('Total Sales')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.gca().get_yaxis().set_major_formatter(FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))
st.pyplot(plt)


def get_top_ten_common_pairs(df):
    pairs_counter = Counter()
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


avgysales = res.groupby(by="YEAR")['SALES'].sum()

col1, col2 = st.columns([1, 2])
with col1:
    st.markdown("#### Yearly Sales")
    st.bar_chart(avgysales)

with col2:
    st.plotly_chart(fig)



sales_by_prodd = res.groupby('PRODUCTLINE')['SALES'].sum().reset_index()

# Creating the pie chart using Altair
c = alt.Chart(sales_by_prodd).mark_arc().encode(
    theta=alt.Theta(field='SALES', type='quantitative'),
    color=alt.Color(field='PRODUCTLINE', type='nominal'),
    tooltip=['PRODUCTLINE', 'SALES']
)




col1, col2, col3 , col4 = st.columns(4)

with col1:

    st.write("Transportation Progress:")
    st.write(sales_by_delaytop5)

with col2:
    st.write("Top 10 Zip Codes by Total Sales:")
    st.write(sales_by_postaltop)
with col3:
    st.write("Top 10 Products by Total Sales:")
    st.altair_chart(c, use_container_width=True)

with col4:
    st.write("Most common pairs of items bought:")
    st.dataframe(comdf)




st.markdown('### Where does select customer rank?')
sales_by_CUSTOMER_NAME = df.groupby('CUSTOMER_NAME')['SALES'].sum().reset_index()

customer_names = sales_by_CUSTOMER_NAME['CUSTOMER_NAME'].tolist()
selected_customer = st.selectbox('Select Customer Name', customer_names)

selected_customer_sales = sales_by_CUSTOMER_NAME[sales_by_CUSTOMER_NAME['CUSTOMER_NAME'] == selected_customer]['SALES'].values[0]

base = alt.Chart(sales_by_CUSTOMER_NAME).mark_bar().encode(
    x=alt.X('SALES:Q', bin=alt.Bin(maxbins=30), title='Sales'),
    y=alt.Y('count()', title='Number of Customers')
)
highlight = alt.Chart(pd.DataFrame({'SALES': [selected_customer_sales]})).mark_rule(color='red').encode(
    x='SALES:Q')

chart = base + highlight
st.altair_chart(chart, use_container_width=True)


