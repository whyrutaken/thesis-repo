# %%
from datetime import datetime
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot as plt
import numpy as np
from preparator import Preparator
from error_metric_calculator import Metrics


class ArimaModel:

    def __init__(self, attribute, test_from_date, test_to_date, horizon, p_values, q_values, d_values):
        # hyperparameters
        self.p_values = p_values
        self.q_values = q_values
        self.d_values = d_values

        self.preparator = Preparator(attribute, test_from_date)
        self.train, self.test = self.preparator.train_test_split_by_date(self.preparator.historical_df,
                                                                         test_from_date=test_from_date)
        self.prediction, self.best_params, self.duration = self.multistep_forecast(test_from_date, test_to_date, horizon=horizon)

        self.individual_error_scores, self.overall_error_scores = Metrics().calculate_errors(self.test, self.prediction)
        self.std_error = self.individual_error_scores.std()

    @staticmethod
    def fit_and_predict(df, test_from_date, horizon, best_params):
        start = datetime.now()
        print("ARIMA fit and predict, test_from_date={}, horizon={}, order={}".format(test_from_date, horizon,
                                                                                      best_params))

        train, test = df.train_test_split_by_date(df.historical_df, test_from_date=test_from_date)
        model = ARIMA(train, order=best_params)
        model = model.fit()
        #     self.plot_model_details(fitted_model)
        prediction = model.forecast(horizon)

        end = datetime.now()
        duration = end - start
        print("Duration: {}".format(duration))
        return prediction

    def grid_search(self, df, test_from_date, horizon):
        best_score = 100000000  # easter egg
        best_order = 0
        for p in self.p_values:
            for q in self.q_values:
                for d in self.d_values:
                    print("ARIMA Grid search with order ({},{},{})".format(p, d, q))
                    prediction = self.fit_and_predict(df, test_from_date, horizon, best_params=(p, d, q))
                    individual_scores, overall_scores = Metrics().calculate_errors(self.test, prediction)
                    if (np.array(overall_scores) < np.array(best_score)).all():
                        best_score = overall_scores
                        best_order = (p, d, q)
        return best_order

    def multistep_forecast(self, test_from_date, test_to_date, horizon):
        start = datetime.now()
        date_range = pd.date_range(test_from_date, test_to_date, freq=str(horizon) + "H")
        best_params = self.grid_search(self.preparator, test_from_date, horizon)
        prediction = []
        for date in date_range:
            prediction = np.append(prediction,
                                   self.fit_and_predict(self.preparator, test_from_date=date, horizon=horizon,
                                                        best_params=best_params))
        end = datetime.now()
        duration = end - start
        print("Total duration of ARIMA multistep forecast: {}".format(duration))
        return self.format_prediction(prediction, self.test), best_params, duration

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

# model = ArimaModel("solar_absolute", test_from_date="2020-01-10 00:00", test_to_date="2020-01-11 10:00", horizon=5)
# printer.print_single_forecast(model.train, model.test, model.prediction)
