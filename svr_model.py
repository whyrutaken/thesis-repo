from datetime import datetime
import pandas as pd
from sklearn.model_selection import GridSearchCV
from preparator import Preparator
from sklearn.svm import SVR
import numpy as np
from error_metric_calculator import Metrics
import tomli


class SVRModel:
    def __init__(self, horizon: int, grid_search: bool):
        # hyperparameters
        attribute, test_from_date, test_to_date, kernel, C, degree, coef0, best_params = self.read_config()
        self.preparator = Preparator(attribute, test_from_date)
        self.y_train, self.y_test = self.preparator.y_train, self.preparator.y_test

        if grid_search:
            self.best_params = self.grid_search(test_from_date, kernel, C, degree, coef0)
        else:
            self.best_params = best_params
            self.prediction, self.duration = self.multistep_forecast(test_from_date=test_from_date,
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
        attribute_, test_from_date_, test_to_date_ = config["attribute"], config["test_from_date"], config[
            "test_to_date"]
        kernel_ = tuple(config["svr"]["kernel"])
        C_ = tuple(config["svr"]["c"])
        degree_ = tuple(config["svr"]["degree"])
        coef0_ = tuple(config["svr"]["coef0"])
        best_params_ = config["svr"]["best_params"]
        return attribute_, test_from_date_, test_to_date_, kernel_, C_, degree_, coef0_, best_params_

    def grid_search(self, test_from_date, kernel, C, degree, coef0):
        start = datetime.now()
        print("SVR Grid search started")
        hyperparameters = dict(kernel=kernel, C=C, degree=degree, coef0=coef0)
        x_train, x_test, y_train, y_test = self.preparator.get_scaled_data(test_from_date=test_from_date)
        model = SVR(verbose=True)  # verbose=True
        cv = [(slice(None), slice(None))]
        rs = GridSearchCV(model, param_grid=hyperparameters, cv=cv, n_jobs=-1)
        rs.fit(x_train, y_train.ravel())
        self.print_end(start, "Total duration of SVR grid search: ")
        return rs.best_params_

    def fit_and_predict(self, test_from_date, horizon, best_params):
        start = datetime.now()
        print("SVR fit and predict, test_from_date={}, horizon={}, best_params={}".format(test_from_date, horizon,
                                                                                          best_params))
        x_train, x_test, y_train, y_test = self.preparator.get_scaled_data(test_from_date=test_from_date)
        model = SVR(kernel=best_params["kernel"], C=best_params["C"], degree=best_params["degree"],
                    coef0=best_params["coef0"], max_iter=-1, shrinking=True,
                    tol=0.001)  # verbose=True
        #   model = SVR(C=10, cache_size=200, coef0=0.0, degree=3, epsilon=0.05, gamma=0.5, kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=True)
        model = model.fit(x_train, y_train.ravel())
        prediction = model.predict(x_test[:horizon])
        inverse_scaled_prediction = self.preparator.inverse_scaler(prediction)
        self.print_end(start, "Duration: ")
        return inverse_scaled_prediction.ravel()

    def multistep_forecast(self, test_from_date, test_to_date, horizon, best_params):
        start = datetime.now()
        date_range = pd.date_range(test_from_date, test_to_date, freq=str(horizon) + "H")
        prediction = []
        for date in date_range:
            prediction = np.append(prediction, self.fit_and_predict(test_from_date=date, horizon=horizon,
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
