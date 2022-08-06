import statsmodels.api as sm
import itertools
import numpy as np
import pandas as pd
import plotly
import time
import os
import plotly.express as px
from flask import Flask, render_template, request ,send_file,render_template

app = Flask(__name__,template_folder='templates')
allowed_extensions = ['csv']

@app.route('/')
def get_file():
    return render_template('upload.html')
	
@app.route('/result', methods = ['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
        f = request.files['file']
        if request.files['file'].filename == '':
            f.seek(0, os.SEEK_END)
            if f.tell() == 0:
                return '<h4>No selected file</h4>'
        else:
            f.save(f.filename)
            if(check_file_extension(f.filename)):
                model(f.filename)
                return render_template('result.html')
            else:
                return "<h4>Oops! Wrong file input.Please input a csv file as input.</h4>"

@app.route('/download')
def download():
    path = "Forecast(2022-26).csv"
    return send_file(path,as_attachment=True)

def check_file_extension(filename):
    return filename.split('.')[-1] in allowed_extensions

def model(file_path):
    df = pd.read_csv(file_path)
    df['Date']=pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df = df.set_index('Date')

    y = df['Spot_price'].resample('MS').mean()
    p = range(0, 2)
    d = range(0, 2)
    q = range(0, 2)

    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y, order = param, seasonal_order = param_seasonal, enforce_stationary = False,enforce_invertibility=False) 
                result = mod.fit()   
            except:
                continue
    model = sm.tsa.statespace.SARIMAX(y, order = (1, 1, 1),
                                    seasonal_order = (1, 1, 0, 12)
                                    )
    result = model.fit(disp=0)
    prediction = result.get_prediction(start = pd.to_datetime('2019-01-01'), dynamic = False)
    prediction_ci = prediction.conf_int()

    y_hat = prediction.predicted_mean
    y_truth = y['2019-01-01':]

    mse = ((y_hat - y_truth) ** 2).median()
    rmse = np.sqrt(mse)

    pred_uc = result.get_forecast(steps = 50)
    pred_ci = pred_uc.conf_int()

    cols=['lower Spot_price','upper Spot_price']
    values=pred_ci[cols].mean()
    pred_ci.insert(2,"Mean",values)
    pred_ci['Mean'] = pred_ci[['lower Spot_price', 'upper Spot_price']].mean(axis=1)
    pred_ci.to_csv('Forecast(2022-26).csv')


if __name__ == '__main__':
   app.run(debug = True)