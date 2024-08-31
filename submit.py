import numpy as np
import pickle as pkl
import pandas as pd
from datetime import datetime
# Define your prediction method here
# df is a dataframe containing timestamps, weather data and potentials
def my_predict( data ):
    with open( "model.pkl", "rb" ) as file:
        model = pkl.load(file)
    data['dates'] = pd.to_datetime(data['Time'],format='%Y-%m-%d %H:%M:%S')
    data['year'] = pd.DatetimeIndex(data['Time']).year
    data['month'] = pd.DatetimeIndex(data['Time']).month
    data['day'] = pd.DatetimeIndex(data['Time']).day
    data['hour'] = pd.DatetimeIndex(data['Time']).hour
    data['minute'] = pd.DatetimeIndex(data['Time']).minute
    data['sec'] = pd.DatetimeIndex(data['Time']).second
    features = ['temp', 'humidity', 'no2op1', 'no2op2', 'o3op1', 'o3op2', 'year', 'month', 'day', 'hour', 'minute', 'sec']
    target_vars = ['OZONE', 'NO2']
    X_test = data[features]
    pred=np.array(model.predict(X_test))
    pred_o3=pred[:,0]
    pred_no2=pred[:,1]
    return ( pred_o3, pred_no2 )