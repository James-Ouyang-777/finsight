import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import yfinance as yf
import yahoo_fin.options as ops
from yahoo_fin.stock_info import get_data
import matplotlib.pyplot as plt

def price_transform(lp, rp):
    output = (rp-lp)/(lp+rp)*100
    return output

td = [0,2,4,9,100]
def test_transform(td):
    out = []
    for i in range(len(td)-1):
        lp = td[i]
        rp = td[i+1]
    out.append(price_transform(lp,rp))
    print(out)
    return out

def get_predictions(new_data, model, num_features=4, look_back=7):
    new_data = new_data.tail(7)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(new_data[['open', 'high', 'low', 'close']])
    print(len(scaled_data))

    # Assuming you have prepared your new data as 'new_data' with the same preprocessing as the training data

    # nd = np.reshape(scaled_data, (scaled_data.shape[0], look_back, num_features))  # Adjust 'look_back' and 'num_features' accordingly
    nd = np.reshape(scaled_data, (1, scaled_data.shape[0], num_features))  # Adjust 'look_back' and 'num_features' accordingly
    # Make predictions using the trained model
    predictions = model.predict(nd)

    predictions = scaler.inverse_transform(np.concatenate((np.zeros((len(predictions), num_features-1)), predictions), axis=1))[:, num_features-1]
    return predictions

"""we can do change transformation for open, close, high, low

transform(a,b) = [a/b-1*100]**(exp) rounded to the nearest multiple of basket size

and volume:volume moving average

the goal is to find patterns in explosive movements to give a ball park prediction
"""
