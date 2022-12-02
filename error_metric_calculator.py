from math import sqrt

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class Metrics:
    @staticmethod
    def rmse_(actual, forecast):
        return np.mean((forecast[0] - actual[0]) ** 2) ** .5

    @staticmethod
    def mae_(actual, forecast):
        return np.mean(np.abs(forecast[0] - actual[0]))

    @staticmethod
    def rmse(actual, forecast):
        return mean_squared_error(actual, forecast, squared=False)

    @staticmethod
    def mae(actual, forecast):
        return mean_absolute_error(actual, forecast)

    @staticmethod
    def r2(actual, forecast):
        return r2_score(actual, forecast)

    def individual_scores(self, actual, forecast):
        rmse_scores = []
        mae_scores = []
        for i in range(len(forecast) - 1):
            rmse_scores.append(self.rmse(actual.iloc[i], forecast.iloc[i]))
            mae_scores.append(self.mae(actual.iloc[i], forecast.iloc[i]))
        df = pd.DataFrame(rmse_scores, columns=["rmse"])
        df["mae"] = mae_scores
        return df

    # Errors of all outputs are averaged with uniform weight.
    def overall_scores(self, actual, forecast):
        rmse_score = self.rmse(actual, forecast)
        mae_score = self.mae(actual, forecast)
        r2_score = self.r2(actual, forecast)
        return pd.DataFrame([[rmse_score, mae_score, r2_score]], columns=["rmse", "mae", "r2"])

    def calculate_errors(self, actual, forecast):
        individual_scores = self.individual_scores(actual[:len(forecast)], pd.DataFrame(forecast))
        overall_scores = self.overall_scores(actual[:len(forecast)], pd.DataFrame(forecast))
        return individual_scores, overall_scores
