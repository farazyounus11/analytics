import streamlit as st

st.set_page_config(layout="wide")

import pandas as pd
import streamlit_pandas as sp
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os

# Create two columns using st.columns
col1, col2 = st.columns(2)

# Content for col1
with col1:
    st.markdown("### The Red sidebar widgets lets users filter for financial requirements in a company. There are many metrics you can filter for. For example, if you increase ~Full_Time_Employees widget, you will only see Amazon & Walmart in the tables. That is because they employ the most")
with col2:
    st.markdown("### Data API: Yahoo fin")
st.write('---')


@st.cache_data
def load_data():
    df = pd.read_csv(file)
    return df




file = "stock2.csv"
df = load_data()

df = df.fillna(df.median())




create_data = {

                "Industry": "multiselect",
                "Sector": "multiselect",
                "Name": "multiselect",
                "Recommendation_Key": "multiselect"}


all_widgets = sp.create_widgets(df, create_data, ignore_columns=["Name", "Long_Business_Summary", "Number_Of_Analyst_Opinions","Shareholder_Rights_Risk","Compensation_Risk", "Open","Fifty_Two_Week_Low", "Current_Price","Fifty_Two_Week_High","Previous_Close","Payout_Ratio","Regular_Market_Volume"])
res = sp.filter_df(df, all_widgets)
st.write('---')

st.markdown('#### Top 15 Companies Sorted by Metrics')


sorted_by_earnings_growth = res.sort_values(by='Earnings_Quarterly_Growth', ascending=False).head(15)[['Name', 'Earnings_Quarterly_Growth']].rename(columns={'Earnings_Quarterly_Growth': 'Quarterly_Growth'})

sorted_by_profit_margins = res.sort_values(by='Profit_Margins', ascending=False).head(15)
sorted_by_profit_margins['Profit_Margins'] = (sorted_by_profit_margins['Profit_Margins'] * 100).round(1)
sorted_by_profit_margins = sorted_by_profit_margins[['Name', 'Profit_Margins']] 

sorted_by_free_cash_flow = res.sort_values(by='Free_Cash_Flow', ascending=False).head(15)[['Name', 'Free_Cash_Flow']]

# Custom CSS for adjusting column width
st.write("""
    <style>
        table {
            width: 100%;
        }
        th, td {
            padding: 8px;
            text-align: left;
        }
        .stDataFrame th:first-child,
        .stDataFrame td:first-child {
            width: 30%; /* Adjust the width of the first column (Name) */
        }
    </style>
""", unsafe_allow_html=True)

# Display in three columns using st.table()
col1, col2, col3 = st.columns(3)

with col1:
    st.write('#### Top Quarterly Earnings Growth')
    st.table(sorted_by_earnings_growth.set_index('Name').style.format({'Quarterly_Growth': '{:.1f}%'}))

with col2:
    st.write('### Sorted by Profit Margins')
    # Formatting Profit_Margins to show one decimal place
    st.table(sorted_by_profit_margins.set_index('Name').style.format({'Profit_Margins': '{:.1f}%'}))

with col3:
    st.write('### Sorted by Free Cash Flow')
    st.table(sorted_by_free_cash_flow.set_index('Name'))




sorted_by_total_cash = res.sort_values(by='Total_Cash', ascending=False).head(15)[['Name', 'Total_Cash']].round({'Total_Cash': 1})
sorted_by_forward_eps = res.sort_values(by='Forward_EPS', ascending=False).head(15)[['Name', 'Forward_EPS']].round({'Forward_EPS': 1})
sorted_by_net_income = res.sort_values(by='Net_Income_To_Common', ascending=False).head(15)[['Name', 'Net_Income_To_Common']].round({'Net_Income_To_Common': 1})




col4, col5, col6 = st.columns(3)

with col4:
    st.markdown('### Sorted by Total Cash')
    st.write(sorted_by_total_cash)

with col5:
    st.markdown('#### Sorted by Forward EPS')
    st.write(sorted_by_forward_eps)

with col6:
    st.markdown('### Sorted by Net Income to Common')
    st.write(sorted_by_net_income)















st.markdown('### Search Up Company Information')
search_query = st.text_input('Enter company name:', '')

# Filter the DataFrame based on the search query
filtered_df = df[df['Name'].str.contains(search_query, case=False, na=False)]

if not search_query:
    st.write('Enter a search query above.')
elif filtered_df.empty:
    st.write('No results found.')
else:
    for index, row in filtered_df.iterrows():
        st.write(f"### {row['Name']}")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write(f"**Industry:** {row['Industry']}")
            st.write(f"**EBITDA_Margins:** {row['EBITDA_Margins']}")
            st.write(f"**Free Cash Flow (in Mil):** {round(row['Free_Cash_Flow'] / 1000000, 2)} Mil")
        
        with col2:
            st.write(f"**Sector:** {row['Sector']}")
            st.write(f"**Revenue_Per_Share:** {row['Revenue_Per_Share']}")
            st.write(f"**Return On Equity:** {row['Return_On_Equity'] * 100:.2f}%")
        
        with col3:
            st.write(f"**Trailing PE:** {round(row['Trailing_PE'], 2)}")
            st.write(f"**Forward PE:** {round(row['Forward_PE'], 2)}")
            st.write(f"**Total Debt (in Mil):** {round(row['Total_Debt'] / 1000000, 2)} Mil")
            
            # Compare performance and calculate percentage difference
        performance_difference = (row['Fifty_Two_Week_Change'] - row['S&P_Fifty_Two_Week_Change']) / row['S&P_Fifty_Two_Week_Change'] * 100

        # Determine color based on performance difference
        color = 'green' if performance_difference > 0 else 'red' if performance_difference < 0 else 'black'

        # Display whether the company beat or did not beat the S&P 500 and by what percentage, with larger and colored text
        if performance_difference > 0:
            st.write(f'<p style="font-size:24px;">{row["Name"]} Outperformed S&P 500 by <span style="color:{color};">{performance_difference:.2f}%</span></p>', unsafe_allow_html=True)
        elif performance_difference < 0:
            st.write(f'<p style="font-size:24px;">{row["Name"]} Underperformed S&P 500 by <span style="color:{color};">{abs(performance_difference):.2f}%</span></p>', unsafe_allow_html=True)
        else:
            st.write(f'<p style="font-size:24px;">{row["Name"]} performed exactly in line with the S&P 500.</p>')

        st.markdown('<p style="font-size:24px;">Business Summary</p>', unsafe_allow_html=True)
        st.write(f"**Summary:** {row['Long_Business_Summary']}")
        st.write('---')























numerical_columns = df.select_dtypes(include=['number']).columns


st.markdown("## Where Does Select Company Rank Amongst All Companies")

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


def format_with_commas(number):
    return '{:,}'.format(number) if isinstance(number, int) else number






