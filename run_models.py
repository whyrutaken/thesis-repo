import pandas as pd
from preparator import Preparator
from svr_model import SVRModel
from arima_model import ArimaModel
from lstm_model import LSTMModel
import printer
import matplotlib.pyplot as plt

import tomli


def get_arima_values(config):
    p_values = (config["arima"]["p1"], config["arima"]["p2"], config["arima"]["p3"])
    q_values = (config["arima"]["q1"], config["arima"]["q2"])
    d_values = (config["arima"]["d1"], config["arima"]["d2"])
    return p_values, q_values, d_values


def get_svr_values(config):
    kernel = (config["svr"]["kernel1"], config["svr"]["kernel2"])
    C = (config["svr"]["c1"], config["svr"]["c2"], config["svr"]["c3"])
    degree = (config["svr"]["degree1"], config["svr"]["degree2"], config["svr"]["degree3"])
    coef0 = (config["svr"]["coef0_1"], config["svr"]["coef0_2"], config["svr"]["coef0_3"])
    gamma = (config["svr"]["gamma1"], config["svr"]["gamma2"])
    return kernel, C, degree, coef0, gamma


def get_lstm_values(config):
    dropout = (config["lstm"]["dropout1"], config["lstm"]["dropout2"], config["lstm"]["dropout3"])
    hidden_layers = (config["lstm"]["hidden_layer1"], config["lstm"]["hidden_layer2"], config["lstm"]["hidden_layer3"],
                     config["lstm"]["hidden_layer4"])
    activation = (config["lstm"]["activation1"], config["lstm"]["activation2"])
    batch_size = config["lstm"]["batch_size"]
    epochs = config["lstm"]["epochs"]
    return dropout, hidden_layers, activation, batch_size, epochs


# %%
if __name__ == '__main__':
    with open("arima_model.toml", mode="rb") as fp:
        config = tomli.load(fp)
    attribute, test_from_date, test_to_date, horizon = config["attribute"], config["test_from_date"], config[
        "test_to_date"], config["horizon"]
    p_values, q_values, d_values = get_arima_values(config)
 #   arima = ArimaModel(attribute=attribute, test_from_date=test_from_date, test_to_date=test_to_date, horizon=horizon,
 #                      p_values=p_values, q_values=q_values, d_values=d_values)

    kernel, C, degree, coef0, gamma = get_svr_values(config)
 #   svr = SVRModel(attribute=attribute, test_from_date=test_from_date, test_to_date=test_to_date, horizon=horizon,
 #                  kernel=kernel, C=C, degree=degree, coef0=coef0, gamma=gamma)

    dropout, hidden_layers, activation, batch_size, epochs = get_lstm_values(config)
    lstm = LSTMModel(attribute=attribute, test_from_date=test_from_date, test_to_date=test_to_date, horizon=horizon,
                     dropout=dropout, hidden_layers=hidden_layers, activation=activation, batch_size=batch_size,
                     epochs=epochs)

# %%
#  lstm = LSTMModel(attribute, test_from_date, test_to_date, horizon)
#   printer.print_single_forecast(lstm.preparator.y_train, lstm.preparator.y_test, lstm.pred)
#  printer.print_double_forecast(lstm.preparator.y_train, lstm.preparator.y_test, lstm.prediction, svr.prediction)

# printer.print_multi_forecast(svr.y_train, svr.y_test, arima.prediction, arima2.prediction, arima3.prediction)
