
import streamlit as st
import pandas as pd
import sys
import os
import numpy as np
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model import StockPricePredictor, StockPrice_Model, scale_features, unscale_predictions
import metrics


st.title('Model Inference Page')

def predict(data):
    stock_model = StockPrice_Model()
    stock_model.load_model() 
    
    data = metrics.calculate_all_features(data)
    
    feature_columns = ['Open','High','Low','Volume','Returns','MA10', 'MA50', 'RSI', 'ATR', 'Volume_Norm', 'Volatility']
   

    features = data[feature_columns]
    
    data = scale_features(features)
    
    predictions = stock_model.predict(data)
    predictions = unscale_predictions(predictions)
    p_data = pd.DataFrame(predictions)
    features['Precited Close'] = p_data.values
    
    return features



uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    dataframe = pd.read_csv(uploaded_file)


if uploaded_file is not None:
    st.header('Dataframe with Predicted values')   
    predictions = predict(dataframe)
    st.dataframe(predictions)