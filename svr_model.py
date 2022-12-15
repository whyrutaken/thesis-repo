from datetime import datetime
from functools import partial
from math import sqrt

import pandas as pd
from keras.losses import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from preparator import Preparator
from sklearn.svm import SVR
import numpy as np
from error_metric_calculator import Metrics
import tomli
from sklearn.model_selection import TimeSeriesSplit
import pickle

class SVRModel:
    def __init__(self, horizon: int, grid_search: bool):
        # hyperparameters
        attribute, train_from_date, test_from_date, test_to_date, kernel, C, degree, gamma, best_params = self.read_config()
        self.preparator = Preparator(attribute, train_from_date=train_from_date, test_from_date=test_from_date)
        self.y_train, self.y_test = self.preparator.y_train, self.preparator.y_test

        if grid_search:
            self.best_params, self.cv_results, self.param_grid, self.best_score = self.grid_search(train_from_date=train_from_date, test_from_date=test_from_date, kernel=kernel, C=C, degree=degree, gamma=gamma)
        else:
            self.best_params = best_params
            self.prediction, self.duration = self.multistep_forecast(train_from_date=train_from_date, test_from_date=test_from_date,
                                                                     test_to_date=test_to_date,
                                                                     horizon=horizon, best_params=self.best_params)

            self.individual_error_scores, self.overall_error_scores = Metrics().calculate_errors(self.preparator.y_test,
                                                                                                 self.prediction)
            self.std_error = self.individual_error_scores.std()

    #   printer.print_error(self.individual_error_scores)
    #   printer.print_single_forecast(self.y_train, self.y_test, self.prediction)

    @staticmethod
    def read_config():
        with open("config.toml", mode="rb") as fp:
            config = tomli.load(fp)
        attribute_, train_from_date_, test_from_date_, test_to_date_ = config["attribute"], config["train_from_date"], config["test_from_date"], config[
            "test_to_date"]
        kernel_ = tuple(config["svr"]["kernel"])
        C_ = tuple(config["svr"]["c"])
        degree_ = tuple(config["svr"]["degree"])
        gamma_ = tuple(config["svr"]["gamma"])
        best_params_ = config["svr"]["best_params"]
        return attribute_, train_from_date_, test_from_date_, test_to_date_, kernel_, C_, degree_, gamma_, best_params_

    def grid_search(self, train_from_date, test_from_date, kernel, C, degree, gamma):
        start = datetime.now()
        print("SVR Grid search started")
        hyperparameters = dict(kernel=kernel, C=C, degree=degree, gamma=gamma)
        x_train, x_test, y_train, y_test = self.preparator.get_scaled_data(test_from_date=test_from_date, train_from_date=train_from_date)
        model = SVR()  # verbose=True
     #   cv = [(slice(None), slice(None))]
        cv = TimeSeriesSplit()

        rs = GridSearchCV(model, param_grid=hyperparameters, n_jobs=-1, verbose=3, return_train_score=True, cv=cv)
        rs.fit(x_train, y_train.ravel())
        self.print_end(start, "Total duration of SVR grid search: ")
        return rs.best_params_, rs.cv_results_, rs.param_grid, rs.best_score_

    def fit_and_predict(self, train_from_date, test_from_date, horizon, best_params):
        start = datetime.now()
        print("SVR fit and predict, train_from_date={}, test_from_date={}, horizon={}, best_params={}".format(train_from_date, test_from_date, horizon,
                                                                                          best_params))
        x_train, x_test, y_train, y_test = self.preparator.get_scaled_data(test_from_date=test_from_date, train_from_date=train_from_date)
        model = SVR(kernel=best_params["kernel"], C=best_params["C"], degree=best_params["degree"],
                    gamma=best_params["gamma"], max_iter=-1, shrinking=True, tol=0.001)  # verbose=True
        #   model = SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.05, gamma=0.5, kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=True)
        model = model.fit(x_train, y_train.ravel())
        prediction = model.predict(x_test[:horizon])
        inverse_scaled_prediction = self.preparator.inverse_scaler(prediction)
        self.print_end(start, "Duration: ")
        return inverse_scaled_prediction.ravel()

    def multistep_forecast(self, train_from_date, test_from_date, test_to_date, horizon, best_params):
        start = datetime.now()
        date_range = pd.date_range(test_from_date, test_to_date, freq=str(horizon) + "H")
        prediction = []
        for date in date_range:
            prediction = np.append(prediction, self.fit_and_predict(train_from_date=train_from_date, test_from_date=date, horizon=horizon,
                                                                    best_params=best_params))
        duration = self.print_end(start, "Total duration of SVR multistep forecast: ")
        return self.format_prediction(prediction), duration

    @staticmethod
    def print_end(start, note):
        end = datetime.now()
        duration = end - start
        print(note, duration)
        return duration

    def format_prediction(self, prediction):
        prediction = pd.Series(prediction)
        prediction.index = self.preparator.y_test[:len(prediction)].index
        prediction.index = pd.DatetimeIndex(prediction.index)
        return prediction

# model = SVRModel("solar_absolute", test_from_date="2020-01-10 00:00", test_to_date="2020-01-11 00:00", horizon=24)


# %%


# scaled_prediction = format_prediction(scaled_prediction[0][:1200], preparator.y_test)

# print_forecast(model.preparator.y_train, model.preparator.y_test, model.prediction3,model.prediction1,"")
# print(model.model.score(x_test, y_test))
