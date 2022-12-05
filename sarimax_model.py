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
from error_metric_calculator import Metrics
from statsmodels.tsa.statespace.sarimax import SARIMAX
import printer
from scipy.stats.stats import pearsonr


class SarimaxModel:

    def __init__(self, attribute, test_from_date, test_to_date, horizon):
        self.preparator = Preparator(attribute, test_from_date)
        self.y_train, self.y_test = self.preparator.train_test_split_by_date(self.preparator.historical_df,
                                                                             test_from_date=test_from_date)
        self.x_train, self.x_test = self.preparator.train_test_split_by_date(self.preparator.weather_df["irrad"],
                                                                             test_from_date=test_from_date)
        #     corr, p = pearsonr(self.y_train.values, self.x_train.values)
        #     print("Correlation coefficient = ", corr, "\nP-value = ", p)
        self.prediction = self.multistep_forecast(test_from_date, test_to_date, horizon=horizon)
        self.individual_scores, self.overall_scores = Metrics().calculate_errors(self.y_test, self.prediction)

    def fit_and_predict(self, df, test_from_date, horizon):
        y_train, y_test = df.train_test_split_by_date(df.historical_df, test_from_date=test_from_date)
        x_train, x_test = df.train_test_split_by_date(df.weather_df["irrad"], test_from_date=test_from_date)

        model = SARIMAX(y_train, exog=x_train, order=(1, 1, 1), seasonal_order=(28, 1, 1, 2),
                        enforce_stationarity=False, enforce_invertibility=False)
        print("fit")
        model = model.fit()
        self.plot_model_details(model)
        prediction = model.forecast(exog=x_test, steps=horizon)
        print("pred")
        return prediction

    def multistep_forecast(self, test_from_date, test_to_date, horizon):
        date_range = pd.date_range(test_from_date, test_to_date, freq=str(horizon) + "H")
        prediction = []
        for date in date_range:
            prediction = np.append(prediction,
                                   self.fit_and_predict(self.preparator, test_from_date=date, horizon=horizon))
        return self.format_prediction(prediction, self.y_test)

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


model = SarimaxModel("solar_absolute", test_from_date="2020-02-01 00:00", test_to_date="2020-02-01 02:00", horizon=1)
printer.print_single_forecast(model.y_train, model.y_test, model.prediction)
