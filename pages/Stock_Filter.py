import streamlit as st

st.set_page_config(layout="wide")

import pandas as pd
import streamlit_pandas as sp
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import os
import altair as alt

# Create two columns using st.columns
col1, col2 = st.columns(2)
with col1:
    st.markdown("### The red sidebar widgets lets users filter for financial requirements in a company. There are many metrics you can filter for. Companies that don't meet requirments are droped!")
with col2:
    st.markdown("### This app also helps users search up information about a companies. This app also lets users see where a company lies in earnings distribution")


@st.cache_data
def load_data():
    df = pd.read_csv(file)
    return df

pd.set_option('display.max_colwidth', 20)
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

st.markdown('#### Top 10 Companies Sorted by Metrics')


sorted_by_earnings_growth = res.sort_values(by='Earnings_Quarterly_Growth', ascending=False).head(10)[['Name', 'Earnings_Quarterly_Growth']].rename(columns={'Earnings_Quarterly_Growth': 'Quarterly_Growth'})
sorted_by_profit_margins = res.sort_values(by='Profit_Margins', ascending=False).head(10)[['Name', 'Profit_Margins']]
sorted_by_free_cash_flow = res.sort_values(by='Free_Cash_Flow', ascending=False).head(10)[['Name', 'Free_Cash_Flow']]
sorted_by_earnings_growth['Name'] = sorted_by_earnings_growth['Name'].apply(lambda x: x[:20])
sorted_by_profit_margins['Name'] = sorted_by_profit_margins['Name'].apply(lambda x: x[:20])
sorted_by_free_cash_flow['Name'] = sorted_by_free_cash_flow['Name'].apply(lambda x: x[:20])

col1, col2, col3 = st.columns(3)

with col1:
    st.write('#### Top Quarterly Earnings Growth')
    st.write(sorted_by_earnings_growth)

with col2:
    st.write('### Sorted by Profit Margins')
    # Formatting Profit_Margins to show one decimal place
    st.write(sorted_by_profit_margins)

with col3:
    st.write('### Sorted by Free Cash Flow')
    st.write(sorted_by_free_cash_flow)




sorted_by_total_cash = res.sort_values(by='Total_Cash', ascending=False).head(15)[['Name', 'Total_Cash']].round({'Total_Cash': 1})
sorted_by_forward_eps = res.sort_values(by='Forward_EPS', ascending=False).head(15)[['Name', 'Forward_EPS']].round({'Forward_EPS': 1})
sorted_by_net_income = res.sort_values(by='Net_Income_To_Common', ascending=False).head(15)[['Name', 'Net_Income_To_Common']].round({'Net_Income_To_Common': 1})
sorted_by_total_cash['Name'] = sorted_by_total_cash['Name'].apply(lambda x: x[:20])
sorted_by_forward_eps['Name'] = sorted_by_forward_eps['Name'].apply(lambda x: x[:20])
sorted_by_net_income['Name'] = sorted_by_net_income['Name'].apply(lambda x: x[:20])


col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('### Sorted by Total Cash')
    st.dataframe(sorted_by_total_cash)  # Don't display index column

with col2:
    st.markdown('### Sorted by Forward EPS')
    st.dataframe(sorted_by_forward_eps)  # Don't display index column

with col3:
    st.markdown('### Sorted by Net Income to Common')
    st.dataframe(sorted_by_net_income)  # Don't display index column














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
        performance_difference = (row['Fifty_Two_Week_Change'] - row['S&P_Fifty_Two_Week_Change']) * 100

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



















st.title("Company Analysis")

# Selectbox for company selection
compnames = df['Name'].unique()
selected_name = st.selectbox("Select a Company", compnames)

# Extract the selected row
selected_row = df[df['Name'] == selected_name].squeeze()

# Columns to display
column_options = ["PEG_Ratio", "Free_Cash_Flow", "Revenue_Growth", "EBITDA_Margins"]

def create_histogram(column_name, selected_row):
    hist = alt.Chart(df).mark_bar().encode(
        alt.X(column_name, bin=alt.Bin(maxbins=50)),
        y='count()'
    ).properties(
        width=300,
        height=300
    )

    rule = alt.Chart(pd.DataFrame({
        'value': [selected_row[column_name]]
    })).mark_rule(color='red').encode(
        x='value:Q'
    )

    text = alt.Chart(pd.DataFrame({
        'value': [selected_row[column_name]],
        'text': [f"{selected_row[column_name]}"]
    })).mark_text(
        align='left',
        baseline='middle',
        dx=7,
        dy=-7,
        color='red'
    ).encode(
        x='value:Q',
        text='text:O'
    )

    return hist + rule + text

# Create columns for the 2x2 grid
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)
with col1:
    chart1 = create_histogram(column_options[0], selected_row)
    st.altair_chart(chart1, use_container_width=True)

with col2:
    chart2 = create_histogram(column_options[1], selected_row)
    st.altair_chart(chart2, use_container_width=True)

with col3:
    chart3 = create_histogram(column_options[2], selected_row)
    st.altair_chart(chart3, use_container_width=True)

with col4:
    chart4 = create_histogram(column_options[3], selected_row)
    st.altair_chart(chart4, use_container_width=True)




