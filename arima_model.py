# %%
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
from preparator import Preparator
from metric_calculator import Metrics


class ArimaModel:

    def __init__(self, attribute):
        self.df = Preparator(attribute)
        self.train, self.test = self.df.train_test_split(train_from_date="2020-08-01",
                                                         test_from_date="2021-01-01 00:00")

        #      self.multi_forecast(["2021-01-01", "2021-01-02", "2021-01-03"], forecast_steps=24)
#        pred1 = self.multi_forecast(train_from_date="2020-08-01",
#                                    forecast_dates=["2021-01-01 00:00", "2021-01-01 01:00", "2021-01-01 02:00",
 #                                                   "2021-01-01 03:00", "2021-01-01 04:00", "2021-01-01 05:00",
 #                                                   "2021-01-01 06:00", "2021-01-01 07:00", "2021-01-01 08:00",
 #                                                   "2021-01-01 09:00", "2021-01-01 10:00", "2021-01-01 11:00",
 #                                                   "2021-01-01 12:00"], forecast_steps=1)
 #       error1, mean_error1 = Metrics().calculate_errors(Metrics.rmse, pred1, self.test)
        self.pred2 = self.multi_forecast(train_from_date="2020-08-01",
                                    forecast_dates=["2021-01-01 00:00", "2021-01-01 03:00", "2021-01-01 06:00",
                                                    "2021-01-01 09:00", "2021-01-01 12:00", "2021-01-01 15:00",
                                                    "2021-01-01 18:00", "2021-01-01 21:00", "2021-01-02 00:00",
                                                    ], forecast_steps=3)
        self.error2, self.mean_error2 = Metrics().calculate_errors(Metrics.rmse, self.pred2, self.test)

    def fit_and_predict_model(self, df, train_from_date, test_from_date, forecast_steps):
        train, test = df.train_test_split(train_from_date, test_from_date)
        model = ARIMA(train, order=(29, 1, 1))
        fitted_model = model.fit()
        #     self.plot_model_details(fitted_model)
        prediction = self.format_prediction(fitted_model.forecast(forecast_steps), test)
        return prediction

    def multi_forecast(self, train_from_date, forecast_dates, forecast_steps):
        prediction = self.fit_and_predict_model(self.df, train_from_date=train_from_date, test_from_date=forecast_dates[0],
                                                forecast_steps=forecast_steps)
        for number in range(len(forecast_dates) - 1):
            prediction = prediction.append(
                self.fit_and_predict_model(self.df, train_from_date, forecast_dates[number + 1],
                                           forecast_steps=forecast_steps))
        self.print_forecast(prediction)
        return prediction

    @staticmethod
    def format_prediction(prediction, test):
        prediction = pd.Series(prediction)
        prediction.index = test[:len(prediction)].index
        prediction.index = pd.DatetimeIndex(prediction.index)
        return prediction

    @staticmethod
    def plot_model_details(fitted_model):
        fitted_model.summary()
        fitted_model.plot_diagnostics()
        plt.show()

    def print_forecast(self, prediction):
        plt.figure(figsize=(15, 5), dpi=100)
        plt.locator_params(axis='x', nbins=5)
        plt.plot(self.train[len(self.train) - 48:], label='training')
        plt.plot(self.test[:len(prediction) + 24], label='actual')
        plt.plot(prediction, label='forecast')

        plt.title('ARIMA: Forecast vs Actual')
        plt.legend(loc='upper right', fontsize=8)
        plt.xlabel('Time')
        plt.ylabel('PV production [Wh]')
        plt.show()


arima = ArimaModel("solar_absolute")

