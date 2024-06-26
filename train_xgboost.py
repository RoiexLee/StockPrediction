import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

from api import data, model

function, symbol, output_size, datatype, api_key, features = "TIME_SERIES_DAILY", "IBM", "full", "csv", "YOUR_API_KEY", ["open", "close", "high", "low"]
data_raw = data.get_data(function, symbol, output_size, datatype, api_key, features)

train_size = 0.8
train_data, valid_data = data.split_data(data_raw, train_size)

scaler = MinMaxScaler(feature_range=(0, 1))
train_data_fit = scaler.fit_transform(train_data)

window_size = 20
x_train, y_train = model.xgboost_input(train_data_fit, window_size)

# model_path = "./static/models/xgboost.json"
# model.xgboost(x_train=x_train, y_train=y_train, model_path=model_path)

valid_data_fit = scaler.transform(valid_data)
x_valid, y_valid = model.xgboost_input(valid_data_fit, window_size)

model = XGBRegressor()
model.load_model("./static/models/xgboost.json")

y_train = scaler.inverse_transform(y_train)

y_pred_train = model.predict(x_train)
y_pred_train = scaler.inverse_transform(y_pred_train)

error = [np.sqrt(mean_squared_error(y_train[:, i], y_pred_train[:, i])) for i in range(len(features))]

data.plot_data(features=features, pattern="fit", actual=y_train, pred=y_pred_train, error=error)

y_valid = scaler.inverse_transform(y_valid)
y_pred_valid = model.predict(x_valid)
y_pred_valid = scaler.inverse_transform(y_pred_valid)

error = [np.sqrt(mean_squared_error(y_valid[:, i], y_pred_valid[:, i])) for i in range(len(features))]

data.plot_data(features=features, pattern="fit", actual=y_valid, pred=y_pred_valid, error=error)