import streamlit as st

stock_info_page = st.Page('stock_info.py',title='Stock Market Information',default=True)
model_info_page = st.Page('model_info.py',title='Model Artifacts')
model_inference_page = st.Page('model_inference.py',title='Model Inference')

pg = st.navigation([stock_info_page, model_info_page,model_inference_page])
pg.run()