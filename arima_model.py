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
import printer


class ArimaModel:

    def __init__(self, attribute, test_from_date, test_to_date, horizon):
        self.preparator = Preparator(attribute, test_from_date)
        self.train, self.test = self.preparator.train_test_split_by_date(self.preparator.historical_df,
                                                                         test_from_date=test_from_date)

        self.prediction = self.multistep_forecast(test_from_date, test_to_date, horizon=horizon)
        self.individual_scores, self.overall_scores = Metrics().calculate_errors(self.test, self.prediction)

    def fit_and_predict(self, df, test_from_date, horizon, order):
        train, test = df.train_test_split_by_date(df.historical_df, test_from_date=test_from_date)
        model = ARIMA(train, order=order)
        model = model.fit()
        #     self.plot_model_details(fitted_model)
        prediction = model.forecast(horizon)
        print("fit_and_pred")
        return prediction

    def grid_search(self, df, test_from_date, horizon):
        p_values = (24, 27, 29)
        q_values = (1, 2)
        d_values = (1, 2)
        best_score = 100000
        best_order = 0
        for p in p_values:
            for q in q_values:
                for d in d_values:
                    prediction = self.fit_and_predict(df, test_from_date, horizon, order=(p, d, q))
                    individual_scores, overall_scores = Metrics().calculate_errors(self.test, prediction)
                    if (np.array(overall_scores) < np.array(best_score)).all():
                        best_score = overall_scores
                        best_order = (p, d, q)
        return best_order

    def multistep_forecast(self, test_from_date, test_to_date, horizon):
        date_range = pd.date_range(test_from_date, test_to_date, freq=str(horizon) + "H")
        self.order = self.grid_search(self.preparator, test_from_date, horizon)
        prediction = []
        for date in date_range:
            prediction = np.append(prediction,
                                   self.fit_and_predict(self.preparator, test_from_date=date, horizon=horizon,
                                                        order=self.order))
        return self.format_prediction(prediction, self.test)

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


model = ArimaModel("solar_absolute", test_from_date="2020-01-10 00:00", test_to_date="2020-01-11 10:00", horizon=5)
printer.print_single_forecast(model.train, model.test, model.prediction)
