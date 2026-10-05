from flask import Flask,redirect,request
import numpy as np 
import pandas as pd 
import joblib

obj=joblib.load('winequality_re.pkl')
model=obj['model']
cols=obj['columns']

app=Flask('__name__')
@app.route('/')
def main():
    return('WINE_QUALITY_testing')
@app.route('/predict')
def predict():
    In=[]
    for i in cols:
        v=request.args.get(f'{i}',type=float)
        In.append(v)
    Input=np.array([In])
    out=model.predict([In])
    return(f'the Wine_re_quality: {out}')
if __name__=='__main__':
    app.run(debug=True)