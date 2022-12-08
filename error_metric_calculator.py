from math import sqrt

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import MinMaxScaler

class Metrics:
    # def __init__(self, prediction, actual):

    @staticmethod
    def rmse(actual, forecast):
        return sqrt(mean_squared_error(actual, forecast))

    @staticmethod
    def mae(actual, forecast):
        return mean_absolute_error(actual, forecast)
    @staticmethod
    def mape(actual, forecast):
        actual = actual + 0.1
        return mean_absolute_percentage_error(actual, forecast)
    @staticmethod
    def r2(actual, forecast):
        return r2_score(actual, forecast)

    @staticmethod
    def rmspe(actual, forecast):
        scaler = MinMaxScaler()
        forecast = scaler.fit_transform(forecast)
        actual = scaler.transform(actual)
        return np.sqrt(np.nanmean(np.square((np.array(forecast) - np.array(actual)) / np.array(actual))))

    def individual_scores(self, actual, forecast):
        rmse_scores = []
        mae_scores = []
   #     rmspe_scores = []
        for i in range(len(forecast)):
            rmse_scores.append(self.rmse(actual.iloc[i], forecast.iloc[i]))
            mae_scores.append(self.mae(actual.iloc[i], forecast.iloc[i]))
  #          rmspe_scores.append(self.rmspe(actual.iloc[i], forecast.iloc[i]))

        df = pd.DataFrame(rmse_scores, columns=["rmse"])
        df["mae"] = mae_scores
  #      df["rmspe"] = rmspe_scores
        df.index = actual.index
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
