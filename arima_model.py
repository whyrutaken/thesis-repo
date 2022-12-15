# %%
#!/usr/bin/env python3
from datetime import datetime
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot as plt
import numpy as np
from preparator import Preparator
from error_metric_calculator import Metrics
import tomli



class ArimaModel:

    def __init__(self, horizon: int, grid_search: bool):
        # hyperparameters
        attribute, train_from_date, test_from_date, test_to_date, p_values, q_values, d_values, best_params = self.read_config()
        self.preparator = Preparator(attribute, train_from_date=train_from_date, test_from_date=test_from_date)
        self.train, self.test = self.preparator.train_test_split_by_date(self.preparator.historical_df,
                                                                         test_from_date=test_from_date)
        if grid_search:
            self.best_params = self.grid_search(df=self.preparator, train_from_date=train_from_date, test_from_date=test_from_date, horizon=horizon, p_values=p_values, q_values=q_values, d_values=d_values)
        else:
            self.best_params = best_params
            self.prediction, self.duration = self.multistep_forecast(train_from_date=train_from_date, test_from_date=test_from_date, test_to_date=test_to_date, horizon=horizon,
                                                                     best_params=best_params)

            self.individual_error_scores, self.overall_error_scores = Metrics().calculate_errors(self.test,
                                                                                                 self.prediction)
            self.std_error = self.individual_error_scores.std()

    @staticmethod
    def read_config():
        with open("config.toml", mode="rb") as fp:
            config = tomli.load(fp)
        attribute_, train_from_date_, test_from_date_, test_to_date_ = config["attribute"], config["train_from_date"], config["test_from_date"], config[
            "test_to_date"]
        p_values_ = tuple(config["arima"]["p"])
        q_values_ = tuple(config["arima"]["q"])
        d_values_ = tuple(config["arima"]["d"])
        best_params_ = tuple(config["arima"]["best_params"])
        return attribute_, train_from_date_, test_from_date_, test_to_date_, p_values_, q_values_, d_values_, best_params_


    def fit_and_predict(self, df, train_from_date, test_from_date, horizon, best_params):
        start = datetime.now()
        print("ARIMA fit and predict, train_from_date={}, test_from_date={}, horizon={}, order={}".format(train_from_date, test_from_date, horizon,
                                                                                      best_params))
        train, test = df.train_test_split_by_date(df.historical_df, test_from_date=test_from_date, train_from_date=train_from_date)
        model = ARIMA(train, order=best_params)
        model.initialize_approximate_diffuse()
        model = model.fit()
        #     self.plot_model_details(fitted_model)
        prediction = model.forecast(horizon)
        self.print_end(start, "Duration: ")
        return prediction

    def grid_search(self, df, train_from_date, test_from_date, horizon, p_values, q_values, d_values):
        start = datetime.now()
        print("ARIMA Grid search started")
        best_score = 100000000  # easter egg
        best_params = 0
        for p in p_values:
            for q in q_values:
                for d in d_values:
                    print("ARIMA Grid search with order ({},{},{})".format(p, d, q))
                    prediction = self.fit_and_predict(df=df, test_from_date=test_from_date, train_from_date=train_from_date, horizon=horizon, best_params=(p, d, q))
                    individual_scores, overall_scores = Metrics().calculate_errors(self.test, prediction)
                    if (np.array(overall_scores) < np.array(best_score)).all():
                        best_score = overall_scores
                        best_params = (p, d, q)

        self.print_end(start, "Total duration of ARIMA grid search: ")
        return best_params

    def multistep_forecast(self, train_from_date, test_from_date, test_to_date, horizon, best_params):
        start = datetime.now()
        date_range = pd.date_range(test_from_date, test_to_date, freq=str(horizon) + "H")
        prediction = []
        for date in date_range:
            prediction = np.append(prediction,
                                   self.fit_and_predict(df=self.preparator, train_from_date=train_from_date, test_from_date=date, horizon=horizon,
                                                        best_params=best_params))
        duration = self.print_end(start, "Total duration of ARIMA multistep forecast: ")
        return self.format_prediction(prediction, self.test), duration

    @staticmethod
    def format_prediction(prediction, test):
        prediction = pd.Series(prediction)
        prediction.index = test[:len(prediction)].index
        prediction.index = pd.DatetimeIndex(prediction.index)
        return prediction

    @staticmethod
    def print_end(start, note):
        end = datetime.now()
        duration = end - start
        print(note, duration)
        return duration

    @staticmethod
    def plot_model_details(fitted_model):
        fitted_model.summary()
        fitted_model.plot_diagnostics()
        plt.show()

# model = ArimaModel("solar_absolute", test_from_date="2020-01-10 00:00", test_to_date="2020-01-11 10:00", horizon=5)
# printer.print_single_forecast(model.train, model.test, model.prediction)
