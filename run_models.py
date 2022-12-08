import pandas as pd
from preparator import Preparator
from svr_model import SVRModel
from arima_model import ArimaModel
from lstm_model import LSTMModel
import printer
import matplotlib.pyplot as plt

# %%
if __name__ == '__main__':
    attribute = "solar_absolute"
    test_from_date = "2020-07-10 00:00"
    test_to_date = "2020-07-12 00:00"
    horizon = 24



    svr = SVRModel(attribute, test_from_date, test_to_date, horizon)



#%%
#    arima = ArimaModel(attribute, test_from_date, test_to_date, horizon)
    lstm = LSTMModel(attribute, test_from_date, test_to_date, horizon)
 #   printer.print_single_forecast(lstm.preparator.y_train, lstm.preparator.y_test, lstm.pred)
    printer.print_double_forecast(lstm.preparator.y_train, lstm.preparator.y_test, lstm.prediction, svr.prediction)

# printer.print_multi_forecast(svr.y_train, svr.y_test, arima.prediction, arima2.prediction, arima3.prediction)
