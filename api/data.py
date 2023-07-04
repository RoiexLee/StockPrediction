import io
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests


def get_data(function, symbol, output_size, datatype, api_key, features):
    api_url = "https://www.alphavantage.co/query?function={}&symbol={}&outputsize={}&datatype={}&apikey={}"
    true_url = api_url.format(function, symbol, output_size, datatype, api_key)
    res = requests.get(true_url)
    df = pd.read_csv(io.StringIO(res.text))
    df.sort_values(by="timestamp", inplace=True)
    return df[features].values


def split_data(data, train_size):
    train_length = int(len(data) * train_size)
    return data[:train_length, :], data[train_length:, :]


def plot_data(features, pattern, actual=None, pred=None, error=None, images_path=None):
    plt.figure()
    for i in range(len(features)):
        plt.title("{} in {}".format(features[i], pattern))
        if actual is not None:
            plt.plot(actual[:, i], color="green", label="actual")
        if pred is not None:
            plt.plot(pred[:, i], color="blue", label="pred")
        if error is not None:
            plt.xlabel("error: {}".format(error[i]))
        plt.legend()
        if images_path is not None:
            pic_path = os.path.join(images_path, "{}.jpg".format(features[i], pattern))
            if os.path.exists(pic_path):
                os.remove(pic_path)
            else:
                os.makedirs(os.path.dirname(pic_path), exist_ok=True)
            plt.savefig(pic_path)
        plt.show()
