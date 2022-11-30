import pandas as pd
from preparator import Preparator
from svr_model import SVRModel
from arima_model import ArimaModel
from lstm_model import LSTMModel
import printer
import matplotlib.pyplot as plt

# %%
if __name__ == '__main__':
    svr = SVRModel("solar_absolute", test_from_date="2020-06-10 00:00", test_to_date="2020-06-16 00:00", horizon=24)

    arima = ArimaModel("solar_absolute", test_from_date="2020-06-10 00:00", test_to_date="2020-06-11 00:00",
                       forecast_steps=24)
    lstm = LSTMModel("solar_absolute", test_from_date="2020-06-10 00:00", test_to_date="2020-06-16 00:00", horizon=24)
    printer.print_double_forecast(lstm.preparator.y_train, lstm.preparator.y_test, lstm.pred, svr.prediction)

# printer.print_multi_forecast(svr.y_train, svr.y_test, arima.prediction, arima2.prediction, arima3.prediction)
