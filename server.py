
from __future__ import print_function
from flask import Flask, render_template,request
import sys

from os import environ
from flask import Flask
import json
import numpy as np
import os

import pandas as pd
import urllib.request as urllib2
import socket
from datetime import datetime
import calendar
from keras import applications
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense, Activation, Reshape, LeakyReLU
from keras.callbacks import CSVLogger
import tensorflow as tf
from scipy.ndimage import imread
import numpy as np
import random
from keras.layers import LSTM , GRU
from keras.layers import Conv1D, MaxPooling1D
from keras import backend as K
import keras
from keras.models import load_model
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
from keras import optimizers
import h5py
from sklearn.preprocessing import MinMaxScaler
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import h5py
from flask import jsonify
from json import encoder
import datetime

import plotly.offline as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
#py.init_notebook_mode(connected=True)

import plotly
import gc
#from predictor_ETH import *


#app = Flask(__name__)
app = Flask(__name__, static_url_path='/static') 
app.config['SECRET_KEY'] = 'I am a cryptocurrency predictor!'

model = load_model('models/LSTM_Poloniex_5minutes_6_1.h5')
modelgru = load_model('models/GRU_Poloniex_5minutes_6_1.h5')

pred_df = pd.DataFrame()
pred_df_past = pd.DataFrame()
pred_df_pastgru = pd.DataFrame()
fig = 0
actual = pd.DataFrame()
graphJSON = 0
counter = 0

def get_data():

	d = datetime.datetime.utcnow()
	unixtime = calendar.timegm(d.utctimetuple())

	unixtime = unixtime /100 *100
	past_unixtime = unixtime- 300*30
	url = 'https://poloniex.com/public?command=returnChartData&currencyPair=USDT_ETH&start='+str(past_unixtime)+'&end=9999999999&period=300'
	openUrl = urllib2.urlopen(url)
	r = openUrl.read()
	openUrl.close()
	d = json.loads(r.decode())
	df = pd.DataFrame(d)
	original_columns=[u'close', u'date', u'high', u'low', u'open',u'volume',u'weightedAverage']
	new_columns = ['Close','Timestamp','High','Low','Open','Volume','Weighted_Average']
	df = df.loc[:,original_columns]
	df.columns = new_columns

	df = df.iloc[-6:,:]
	datatimes = df.Timestamp
	#datas = df.Close

	wa = df.Weighted_Average
	datas1 = np.array(df.Close)
	datas2 = np.array(df.Weighted_Average)
	datas = np.column_stack((datas1,datas2))
	datas.shape

	return datas ,datatimes ,wa


@app.route('/plot')

def plot():

	datas ,datatimes, wa = get_data()
	global pred_df_past
	global pred_df_pastgru
	global fig
	global actual
	global graphJSON


	temp_actual = pd.DataFrame()	
	temp_actual['price'] = datas[:,0]  #keep the close actual price
	temp_actual['times'] = pd.to_datetime(datatimes.values,unit='s') #keep the timestamps
	temp_actual['WA'] = wa.values #keep the weighted average

	actual = actual.append(temp_actual,ignore_index=True)

	actual = actual.drop_duplicates(subset='times',keep='last')

	with h5py.File(''.join(['ethereum2015to2018-test.h5']), 'r') as hf:
	    original_datas = hf['original_datas'].value

	scaler = MinMaxScaler()
	scaler.fit(original_datas[:,0].reshape(-1,1))
	datas = scaler.transform(datas)
	datas = datas[None,:,:]

	step_size = datas.shape[1]
	batch_size= 843
	nb_features = datas.shape[2]
	epochs = 1
	output_size=1
	units= 50
	second_units=30

	predicted = model.predict(datas)
	predictedgru = modelgru.predict(datas)

	gc.collect()

####LSTM Prediction####
	predicted_inverted = scaler.inverse_transform(predicted)
	output={}
	output['prediction'] = list(predicted_inverted.reshape(-1))
	datatimes=np.array(datatimes)

####GRU Prediction####
	predicted_invertedgru = scaler.inverse_transform(predictedgru)
	outputgru={}
	outputgru['prediction'] = list(predicted_invertedgru.reshape(-1))

	outputtimes = []
	times = datatimes[-1]

	for i in range(output_size) :
	    
	    if (i == 0):

	        outputtimes.append(times + 300)

	    else:

	        temp = outputtimes[i-1] + 300
	        outputtimes.append(temp)        


	output = pd.DataFrame(output)
	output['times'] = list(outputtimes)
	
	#output = pd.DataFrame(output)
	#output['times'] = output['times'] 
	output.times = pd.to_datetime(output.times,unit='s')


####GRU### output times
	outputgru = pd.DataFrame(outputgru)
	outputgru['times'] = output.times


	actual.times = pd.to_datetime(actual.times,unit='s')

	print ('done', file = sys.stderr)
	
	#output['prediction'] = output['prediction']

####LSTM past prediction to append####
	pred_df_past = pred_df_past.append(output)
	pred_df_past = pred_df_past.drop_duplicates(subset='times',keep='last')
	output = pred_df_past

####GRU past prediction to append####
	pred_df_pastgru = pred_df_pastgru.append(outputgru)
	pred_df_pastgru = pred_df_pastgru.drop_duplicates(subset='times',keep='last')
	outputgru = pred_df_pastgru

	actual_chart = go.Scatter(x = actual['times'], y = actual['price'], name= 'Actual Price')
	lstm_predict_chart = go.Scatter(x = output['times'], y = output['prediction'], name= 'LSTM Prediction Price',mode='lines+markers')
	gru_predict_chart = go.Scatter(x = outputgru['times'], y = outputgru['prediction'], name= 'GRU Prediction Price',mode='lines+markers')
	layout = go.Layout(
	    title='Ethereum Real time Prediction',
	    xaxis=dict(
	        title='Time(UTC)',
	        titlefont=dict(
	            family='Courier New, monospace',
	            size=18,
	            color='#7f7f7f'
	        )
	    ),
	    yaxis=dict(
	        title='ETH(USD)',
	        titlefont=dict(
	            family='Courier New, monospace',
	            size=18,
	            color='#7f7f7f'
	        )
	    )
	)
	data = [actual_chart,lstm_predict_chart,gru_predict_chart]
	fig = go.Figure(data=data, layout=layout)

	graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

	return graphJSON


@app.route('/predictor')

def api_predict():
	print ('start', file = sys.stderr)

	
	global pred_df
	global pred_df_past
	global pred_df_pastgru
	global fig
	global actual
	global graphJSON
	global counter

	if (counter == 0):
		graphJSON = plot()  #call the plot function once
		counter = 1

	return render_template('index.html', graphJSON=graphJSON)

@app.route('/')
def index():

	website = api_predict()

	return website
# app.run(port="8080")
if __name__ == '__main__':
	# HOST = environ.get('HOST', 'localhost')
	HOST=socket.gethostname()
	try:
		PORT = int(environ.get('PORT','5555'))
	except ValueError:
		PORT = 5555
	app.run('0.0.0.0', PORT, threaded=False)
