from keras import models
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler

from api import data, model

function, symbol, output_size, datatype, api_key, features = "TIME_SERIES_DAILY_ADJUSTED", "IBM", "full", "csv", "", ["open", "close", "high", "low"]
data_raw = data.get_data(function, symbol, output_size, datatype, api_key, features)

train_size = 0.8
train_data, valid_data = data.split_data(data_raw, train_size)

scaler = MinMaxScaler(feature_range=(0, 1))
train_data_fit = scaler.fit_transform(train_data)

window_size = 20

x_train, y_train = model.lstm_input(train_data_fit, window_size)

units, dropout, loss, optimizer, model_path, epochs = 100, 0.2, "mse", "adam", "./static/models/lstm.h5", 100
model.lstm(units=units, dropout=dropout, loss=loss, optimizer=optimizer, x_train=x_train, y_train=y_train, model_path=model_path, epochs=epochs)

valid_data_fit = scaler.transform(valid_data)
x_valid, y_valid = model.lstm_input(valid_data_fit, window_size)

model_lstm = models.load_model("./static/models/lstm.h5")
y_pred = model_lstm.predict(x_valid)

error = [mean_absolute_percentage_error(valid, pred) for valid, pred in zip(y_valid, y_pred)]

data.plot_data(features=features, pattern="fit", actual=y_valid, pred=y_pred, error=error)
