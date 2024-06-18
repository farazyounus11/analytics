import os
import streamlit as st
import pandas as pd
import streamlit_pandas as sp
from itertools import combinations
from collections import Counter


st.set_page_config(layout="wide")
st.title('Revenue Analytics App by Faraz')

@st.cache_data
def read_csv(filename):
    df1 = pd.read_csv(filename)
    return df1

def main():
    st.markdown("## Use Sidebar to Select Month of Analysis and Then Filter!")

    current_directory = os.getcwd()
    # st.write("Current Directory:", current_directory)

    file_list = [f for f in os.listdir() if os.path.isfile(f) and f.endswith("_2019.csv")]
    selected_file = st.selectbox("Select Which Month You Want to View", file_list)

    
    if selected_file:
        df1 = read_csv(selected_file)
        df1 = df1.dropna()

        # Convert columns to float
        columns_to_convert = ["Quantity Ordered", "Price Each"]
        for col in columns_to_convert:
            df1[col] = pd.to_numeric(df1[col], errors='coerce')

        # Convert Order Date to datetime and create Day and Hour columns
        try:
            df1['Order Date'] = pd.to_datetime(df1['Order Date'], format='%m/%d/%y %H:%M', errors='coerce')
            df1['Day'] = df1['Order Date'].dt.day
            df1['Hour'] = df1['Order Date'].dt.hour
            df1 = df1.drop(columns=['Order Date'])

        except pd.errors.ParserError:
            st.error('Error: Unknown date format detected in the "Order Date" column.')

        # Create 'Total' column
        df1['Total'] = df1['Quantity Ordered'] * df1['Price Each']
        
        # Reorder columns
        df1 = df1[['Order ID', 'Product', 'Quantity Ordered', 'Price Each', 'Total', 'Day', 'Hour', 'Purchase Address']]

        return df1



# Run the app
if __name__ == '__main__':
    df1 = main()

create_data = {"Order ID": "text",
               "Product": "multiselect",
               "Purchase Address": "text",
               "Order Date": "slider"}
all_widgets = sp.create_widgets(df1, create_data)

# Filter DataFrame
df = sp.filter_df(df1, all_widgets)

st.header("This is Filtered Data")
st.write(df)

grouped_data_address = df.groupby('Purchase Address')['Total'].sum()
top_15_address = grouped_data_address.nlargest(15)
top_15_address_df = top_15_address.reset_index()
top_15_address_df['Total'] = top_15_address_df['Total'].round()

# Group by 'Product' and sum 'Total'
grouped_data_product = df.groupby('Product')['Total'].sum()
top_15_product = grouped_data_product.nlargest(15)
top_15_product_df = top_15_product.reset_index()
top_15_product_df['Total'] = top_15_product_df['Total'].round()

# Group by 'Product' and sum 'Quantity Ordered'
grouped_data_quantity = df.groupby('Product')['Quantity Ordered'].sum()
top_15_quantity = grouped_data_quantity.nlargest(15)
top_15_quantity_df = top_15_quantity.reset_index()
top_15_quantity_df['Quantity Ordered'] = top_15_quantity_df['Quantity Ordered'].round()
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('## Top 15 Purchase Addresses')
    st.write(top_15_address_df)

with col2:
    st.markdown('## Top 15 Products by Total Purchase')
    st.write(top_15_product_df)

with col3:
    st.markdown('## Top 15 Products by Quantity Ordered')
    st.write(top_15_quantity_df)



grouped_data_hour = df.groupby('Hour')['Total'].sum()
grouped_data_day = df.groupby('Day')['Total'].sum()


col1, col2 = st.columns(2)

with col1:
    st.markdown('## Total Purchase Amount by Day')
    st.line_chart(grouped_data_day)

with col2:
    st.markdown('## Total Purchase Amount by Hour')  # You can add another chart here if needed
    st.line_chart(grouped_data_hour)


grouped_data_order = df.groupby('Order ID')['Product'].agg(list)
pair_counter = Counter()

for products in grouped_data_order:
    product_pairs = combinations(products, 2)
    pair_counter.update(product_pairs)

top_pairs = pair_counter.most_common(15)
top_pairs_df = pd.DataFrame(top_pairs, columns=['Product Pair', 'Count'])

# Separate products bought before and after 12:00 PM
products_before_12 = df[df['Hour'] < 12]['Product']
products_after_12 = df[df['Hour'] >= 12]['Product']

# Count occurrences of each product
top_10_before_12 = products_before_12.value_counts().head(10)
top_10_after_12 = products_after_12.value_counts().head(10)

# Display results in three columns using Streamlit
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('### Top 10 Bought Before 12:00')
    st.write(top_10_before_12)

with col2:
    st.markdown('### Top 10 Bought After 12:00')
    st.write(top_10_after_12)

with col3:
    st.markdown('### Top 15 Products Bought Together')
    st.write(top_pairs_df)
