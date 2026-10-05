import numpy as np
import pandas as pd
import streamlit as st
import joblib
import sklearn


obj=joblib.load('winequality_re.pkl')
model=obj['model']
col=obj['columns']

st.title('Wine_red_quality')
In=[]
for i in col:
    v=st.number_input(f'Enter {i} value=')
    In.append(v)
if st.button('click'):
    out=model.predict([In])  
    st.success(f'The wine_red_quality is__ :{out}')

# python -m ven ml_linear
