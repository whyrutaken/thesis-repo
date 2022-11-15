# %%
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np


class PersistenceModel:
    def __init__(self, attribute, from_date, to_date):
        self.master_df = pd.read_csv("extracted-data/master-df.csv")
        self.df = self.set_attribute_and_dates(attribute, from_date, to_date)
        #     self.plot_initial(attribute)
        self.prediction = self.fit_and_predict_model(self.df)
        self.rmse, self.mae, self.mse = self.calculate_errors(self.prediction[:24], self.test_y[:24])
        self.average_rmse, self.average_mae, self.average_mse = self.calculate_average_errors(self.rmse, self.mae,
                                                                                              self.mse)
        self.plot()

    def plot_initial(self, attribute):
        self.master_df.loc[:, attribute].plot()
        plt.show()

    def set_attribute_and_dates(self, attr, from_date, to_date, ):
        return self.master_df[from_date:to_date].loc[:, attr]

    def create_lagged_dataset(self, df) -> pd.DataFrame:
        values = pd.DataFrame(df)
        df = pd.concat([values.shift(1), values], axis=1)
        df.columns = ['t-1', 't+1']
        df.index = pd.DatetimeIndex(df.index)
        print(df.head(5))
        return df

    def train_test_split(self, df):
        train_size = int(len(df) * 0.6)
        train, test = df[1:train_size], df[train_size:]
        self.train_X, self.train_y = pd.Series(train.iloc[:, 0]), pd.Series(train.iloc[:, 1])
        self.test_X, self.test_y = pd.Series(test.iloc[:, 0]), pd.Series(test.iloc[:, 1])

    @staticmethod
    def model_persistence(x):
        return x

    def fit_and_predict_model(self, df):
        self.lagged_set = self.create_lagged_dataset(df)
        self.train_test_split(self.lagged_set)
        return self.predict()

    def predict(self):
        predictions = list()
        for x in self.test_X:
            yhat = self.model_persistence(x)
            predictions.append(yhat)
        return pd.Series(predictions, index=self.test_y.index)

    def calculate_errors(self, forecast, test):
        rmse = []
        mae = []
        mse = []
        for i in range(len(forecast) - 1):
            rmse.append(self.rmse(forecast.iloc[i], test[i]))
            mae.append(self.mae(forecast.iloc[i], test[i]))
            mse.append(self.mse(forecast.iloc[i], test[i]))
        return rmse, mae, mse

    def calculate_average_errors(self, rmse, mae, mse):
        return pd.Series(rmse).mean(), pd.Series(mae).mean(), pd.Series(mse).mean()

    def rmse(self, forecast, actual):
        return np.mean((forecast - actual) ** 2) ** .5

    def mae(self, forecast, actual):
        return np.mean(np.abs(forecast - actual))

    def mse(self, forecast, actual):
        return np.mean((forecast - actual) ** 2)

    def plot(self):
        plt.figure(figsize=(12, 5), dpi=100)
        plt.locator_params(axis='x', nbins=5)

        plt.plot(self.train_y, label='training')
        plt.plot(self.test_y, color="green", label='actual')
        plt.plot(self.prediction, color="red", label="forecast")
        plt.title('Persistence model: Forecast vs Actual')
        plt.legend(loc='upper right', fontsize=8)
        plt.xlabel('Time')
        plt.ylabel('PV production [Wh]')
        plt.show()


solar_model = PersistenceModel("solar_absolute", "2021-12-30", "2022-01-04")
demand_model = PersistenceModel("demand_absolute", "2021-12-30", "2022-01-04")
