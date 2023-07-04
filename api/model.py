import numpy as np
from keras import Sequential
from keras.layers import LSTM, Dropout, Dense
from xgboost import XGBRegressor


def lstm_input(data, window_size):
    x, y = [], []
    for i in range(len(data) - window_size):
        x.append(data[i:i + window_size, :])
        y.append(data[i + window_size, :])
    return np.array(x), np.array(y)


def lstm(units, dropout, loss, optimizer, x_train, y_train, model_path, epochs):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Dropout(dropout))
    model.add(LSTM(units=units, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(units=units))
    model.add(Dropout(dropout))
    model.add(Dense(units=y_train.shape[1]))
    model.compile(loss=loss, optimizer=optimizer)
    model.fit(x_train, y_train, epochs=epochs)
    model.save(model_path)


def lstm_future(data, window_size, days, model):
    future = data[-window_size:, :]
    for i in range(days):
        x = future[i:i + window_size:, :]
        x = np.expand_dims(x, axis=0)
        y = model(x)
        future = np.concatenate((future, y), axis=0)
    return future


def xgboost_input(data, window_size):
    x, y = [], []
    for i in range(len(data) - window_size):
        x.append(data[i:i + window_size, :].flatten())
        y.append(data[i + window_size, :])
    return np.array(x), np.array(y)


def xgboost(x_train, y_train, model_path):
    model = XGBRegressor()
    model.fit(x_train, y_train)
    model.save_model(model_path)


def xgboost_future(data, window_size, days, model):
    future = data[-window_size:, :]
    for i in range(days):
        x = future[i:i + window_size:, :].flatten()
        y = model(x)
        np.concatenate((future, y), axis=0)
    return future
