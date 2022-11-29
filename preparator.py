import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import datetime


class Preparator:
    def __init__(self, attribute, test_from_date):
        self.scaler_x = 0
        self.scaler_y = 0
        self.historical_df = self.load_historical_data(attribute)
        self.weather_df = self.load_weather_data(attribute)
        self.x_train, self.x_test = self.train_test_split_by_date(self.weather_df, test_from_date=test_from_date)
        self.y_train, self.y_test = self.train_test_split_by_date(self.historical_df, test_from_date=test_from_date)

    def get_scaled_data(self, test_from_date):
        x_train, x_test = self.train_test_split_by_date(self.weather_df, test_from_date=test_from_date)
        y_train, y_test = self.train_test_split_by_date(self.historical_df, test_from_date=test_from_date)
        y_train = np.asarray(y_train).reshape(-1, 1)
        y_test = np.asarray(y_test).reshape(-1, 1)
        self.scaler_x, x_train, x_test = self.scaler(x_train, x_test)
        self.scaler_y, y_train, y_test = self.scaler(y_train, y_test)
        return x_train, x_test, y_train, y_test

    def get_data(self):
        return self.x_train, self.x_test, self.y_train, self.y_test

    @staticmethod
    def load_historical_data(attribute, from_date="2020-01-01 00:00", to_date="2022-02-28 23:00") -> pd.DataFrame:
        master_df = pd.read_csv("extracted-data/master-df.csv")
        master_df.index = pd.DatetimeIndex(master_df.index)
        return master_df[from_date:to_date].loc[:, attribute]

    @staticmethod
    def load_weather_data(attribute, from_date="2020-01-01 00:00", to_date="2022-02-28 23:00"):
        radiation_data = pd.read_csv("56.21830411660761_10.146653736350844_Solcast_PT60M.csv",
                                     parse_dates=["PeriodEnd"], index_col="PeriodEnd")
        weather_data = pd.read_csv("historical-weather.csv", parse_dates=["dt_iso"], index_col="dt_iso")
        radiation_data.index = pd.DatetimeIndex(radiation_data.index)
        radiation_data = radiation_data[from_date:to_date]

        weather_data = weather_data.iloc[:len(radiation_data)]
        weather_df = weather_data[["temp", "humidity", "pressure", "wind_speed", "clouds_all"]]
        weather_df.index = radiation_data.index
        weather_df["irrad"] = radiation_data["Ghi"]
        weather_df["hour"] = weather_df.index.hour
        weather_df["season"] = (weather_df.index.month % 12 + 3) // 3
        if attribute == "demand_absolute":
            weather_df["isweekend"] = weather_df.index.dayofweek > 4
            weather_df["isweekend"] = weather_df["isweekend"].astype(int)
        return weather_df

    @staticmethod
    def scaler(train, test):
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train)
        test_scaled = scaler.transform(test)
        return scaler, train_scaled, test_scaled

    def inverse_scaler(self, prediction):
        inversed_pred = self.scaler_y.inverse_transform(prediction.reshape(-1, 1))
        return inversed_pred

    @staticmethod
    def train_test_split(df, train_size=17544):
        train, test = train_test_split(df, shuffle=False, train_size=train_size)
        train.index = pd.DatetimeIndex(train.index)
        test.index = pd.DatetimeIndex(test.index)
        return train, test

    @staticmethod
    def train_test_split_by_date(df, train_from_date="2020-01-01 00:00", test_from_date="2022-01-01 00:00"):
        # for arimamodel these were pd.series
        train = pd.DataFrame(df[train_from_date:test_from_date], index=df[train_from_date:test_from_date].index)
        test = pd.DataFrame(df[test_from_date:], index=df[test_from_date:].index)
        train.index = pd.DatetimeIndex(train.index, freq=train.index.inferred_freq)
        test.index = pd.DatetimeIndex(test.index, freq=train.index.inferred_freq) 
        return train, test


preparator = Preparator("solar_absolute", "2022-01-01")
