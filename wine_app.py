import numpy as np
import pandas as pd
import streamlit as st
import joblib
import sklearn


obj=joblib.load('California.joblib')
model=obj['model']
col=obj['columns']

st.title('California app')
In=[]
for i in col:
    v=st.number_input(f'Enter {i} value=')
    In.append(v)
if st.button('click'):
    out=model.predict([In])  
    st.success(f'The median House value is :{out}')

# python -m ven ml_linear