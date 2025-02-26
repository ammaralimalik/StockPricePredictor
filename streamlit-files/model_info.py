
import streamlit as st
import pandas as pd
import sys
import os
from PIL import Image
import numpy as np
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


st.title('Model Artifacts & Metrics')

metrics = pd.read_csv('/Users/ammarmalik/Desktop/ResumeProjects/StockPricePredictor/model_artifacts/model_metrics.csv')

st.dataframe(metrics)

col1, col2 = st.columns(2)

st.subheader('Predicted vs Actual Values')
image = Image.open('model_artifacts/actual_vs_predicted.png')
st.image(image,use_container_width=True)

st.subheader('Regression Plot')
image = Image.open('model_artifacts/regression_plot.png')
st.image(image,use_container_width=True)

st.subheader('Residual Distribution Plot')
image = Image.open('model_artifacts/residual_distribution.png')
st.image(image,use_container_width=True)

st.subheader('Residual Plot')
image = Image.open('model_artifacts/residual_plot.png')
st.image(image,use_container_width=True)
