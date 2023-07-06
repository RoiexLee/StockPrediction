import os.path

import numpy as np
from flask import Flask, request, jsonify, render_template
from keras import models
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

from api import data, model


class Struct:

    def __init__(self):
        # 基本配置
        self.lstm_path = "./static/models/lstm.h5"
        self.xgboost_path = "./static/models/xgboost.json"
        self.images_path = "./static/images"

        # 更新图片
        self.init_count = 0
        self.fit_count = 0
        self.predict_count = 0

        # 爬取数据
        self.function = "TIME_SERIES_DAILY_ADJUSTED"
        self.symbol = None
        self.output_size = "full"
        self.datatype = "csv"
        self.api_key = None
        self.features = ["open", "close", "high", "low"]

        # 准备
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.window_size = 20

        # 拟合
        self.data_raw = None
        self.model_name = None
        self.model = None
        self.data_fit = None
        self.data_input = None
        self.data_output = None
        self.data_pred = None
        self.error = None

        # 预测
        self.days = 2
        self.data_future = None


app = Flask(__name__, static_url_path="/static")
struct = Struct()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/raw", methods=["POST"])
def raw():
    struct.api_key = request.json["api_key"]
    struct.symbol = request.json["symbol"]

    try:
        struct.data_raw = data.get_data(function=struct.function,
                                        symbol=struct.symbol,
                                        output_size=struct.output_size,
                                        datatype=struct.datatype,
                                        api_key=struct.api_key,
                                        features=struct.features)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    struct.init_count += 1

    pattern = "raw"
    data.plot_data(features=struct.features, pattern=pattern,
                   actual=struct.data_raw,
                   images_path=os.path.join(struct.images_path, struct.symbol, pattern))

    re = {"{}_{}".format(pattern, feature): os.path.join(struct.images_path, struct.symbol, pattern, "{}.jpg?v={}".format(feature, struct.init_count))
          for feature in struct.features}
    re["status"] = "获取数据成功"

    return jsonify(re), 200


@app.route("/fit", methods=["POST"])
def fit():
    if struct.data_raw is None:
        return jsonify({"error": "未获得数据"}), 400

    struct.model_name = request.json["model_name"]
    struct.data_fit = struct.scaler.fit_transform(struct.data_raw)

    if struct.model_name == "LSTM":
        struct.model = models.load_model(struct.lstm_path)
        struct.data_input, struct.data_output = model.lstm_input(struct.data_fit, struct.window_size)
        struct.data_pred = struct.scaler.inverse_transform(struct.model(struct.data_input))
    elif struct.model_name == "XGBoost":
        struct.model = XGBRegressor()
        struct.model.load_model(struct.xgboost_path)
        struct.data_input, struct.data_output = model.xgboost_input(struct.data_fit, struct.window_size)
        struct.data_pred = struct.scaler.inverse_transform(struct.model.predict(struct.data_input))
    else:
        return jsonify({"error": "不支持的模型"}), 400

    struct.error = [np.sqrt(mean_squared_error(struct.data_raw[struct.window_size:, i], struct.data_pred[:, i])) for i in range(len(struct.features))]

    struct.fit_count += 1

    pattern = "fit"
    data.plot_data(features=struct.features, pattern=pattern,
                   actual=struct.data_raw[struct.window_size:, :], pred=struct.data_pred, error=struct.error,
                   images_path=os.path.join(struct.images_path, struct.symbol, pattern))
    re = {"{}_{}".format(pattern, feature): os.path.join(struct.images_path, struct.symbol, pattern, "{}.jpg?v={}".format(feature, struct.fit_count))
          for feature in struct.features}
    re["status"] = "训练完成"

    return jsonify(re), 200


@app.route("/future", methods=["POST"])
def future():
    try:
        days = int(request.json["days"])
        if days <= 0:
            return jsonify({"error": "预测天数必须大于 0"}), 400
    except ValueError:
        return jsonify({"error": "必须是一个整数值"}), 400

    struct.days = days
    if struct.model_name == "LSTM":
        future_fit = model.lstm_future(data=struct.data_fit, model=struct.model, window_size=struct.window_size, days=struct.days)
    elif struct.model_name == "XGBoost":
        future_fit = model.xgboost_future(data=struct.data_fit, model=struct.model, window_size=struct.window_size, days=struct.days)
    else:
        return jsonify({"error": "不支持的模型"}), 400
    struct.data_future = struct.scaler.inverse_transform(future_fit)

    struct.predict_count += 1

    pattern = "future"
    data.plot_data(features=struct.features, pattern=pattern,
                   pred=struct.data_future,
                   images_path=os.path.join(struct.images_path, struct.symbol, pattern))
    re = {"{}_{}".format(pattern, feature): os.path.join(struct.images_path, struct.symbol, pattern, "{}.jpg?v={}".format(feature, struct.predict_count))
          for feature in struct.features}
    re["status"] = "预测完毕"

    return jsonify(re), 200


if __name__ == "__main__":
    app.run()
