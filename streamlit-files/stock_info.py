import numpy as np
import pandas as pd
import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import metrics

st.title('Stock Predictor Dashboard')

st.header('Stock Information Page')

data = pd.read_csv('/Users/ammarmalik/Desktop/ResumeProjects/StockPricePredictor/data/indexProcessed.csv')
stock_data = metrics.calculate_all_features(data)
stock_symbols = stock_data.index.get_level_values('Index').unique()

option = st.selectbox('Please select a stock symbol to view analytics for', stock_symbols)

chart_data = stock_data.loc[option]

open_data = chart_data['Open']

st.header('Pricing Chart')
st.line_chart(open_data, color=[256,0,0], x_label='Dates', y_label='Price', use_container_width=True)


volatility_data = chart_data['Volatility']
st.header('Volatility')
st.line_chart(volatility_data, color=[256,0,0], x_label='Dates', y_label='Price', use_container_width=True)



rsi_data = chart_data['RSI']
st.header('RSI Chart')
st.line_chart(rsi_data, color=[256,0,0], x_label='Dates', y_label='Price', use_container_width=True) 