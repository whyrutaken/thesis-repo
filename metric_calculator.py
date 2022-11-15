import pandas as pd
import numpy as np


class Metrics:
    @staticmethod
    def rmse(forecast, actual):
        return np.mean((forecast - actual) ** 2) ** .5

    @staticmethod
    def mae(forecast, actual):
        return np.mean(np.abs(forecast - actual))

    @staticmethod
    def mse(forecast, actual):
        return np.mean((forecast - actual) ** 2)

    @staticmethod
    def multi_step_error(function, forecast, actual):
        error = []
        for i in range(len(forecast) - 1):
            error.append(function(forecast.iloc[i], actual[i]))
        return error

    @staticmethod
    def calculate_average_errors(errors):
        return pd.Series(errors).mean()

    def calculate_errors(self, error_function, forecast, actual):
        error = self.multi_step_error(error_function, forecast, actual)
        mean_error = self.calculate_average_errors(error)
        return error, mean_error
