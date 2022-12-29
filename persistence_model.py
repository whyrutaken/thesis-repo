# %%
import pandas as pd
import numpy as np
from preparator import Preparator
from error_metric_calculator import Metrics
import printer


class PersistenceModel:
    def __init__(self, attribute, train_from_date, test_from_date, test_to_date):
        self.attribute, self.train_from_date, self.test_from_date, self.test_to_date = attribute, train_from_date, test_from_date, test_to_date
     #   self.attribute, train_from_date, test_from_date, test_to_date = self.read_config(config)
        self.preparator = Preparator(self.attribute, train_from_date=self.train_from_date, test_from_date=self.test_from_date)
        self.train, self.test = self.preparator.y_train, self.preparator.y_test
        self.prediction = self.multistep_forecast(self.test_from_date, self.test_to_date)
        self.individual_scores, self.overall_scores = Metrics().calculate_errors(self.test, self.prediction)
        self.std_error = self.individual_scores.std()

    @staticmethod
    def read_config(config):
        attribute_, train_from_date_, test_from_date_, test_to_date_ = config["attribute"], config["train_from_date"], \
                                                                       config["test_from_date"], config[
                                                                           "test_to_date"]
        return attribute_, train_from_date_, test_from_date_, test_to_date_

    def create_lagged_dataset(self, df) -> pd.DataFrame:
        values = pd.DataFrame(df)
        if self.attribute == "solar_absolute":
            shift = 24
        else:
            shift = 168
        df = pd.concat([values.shift(shift), values], axis=1).dropna()
        df.columns = ['t-' + str(shift), 't+1']
        df.index = pd.DatetimeIndex(df.index)
        return df

    def fit_and_predict(self, df, test_from_date):
        train, test = df.train_test_split_by_date(df.historical_df, test_from_date=test_from_date)
        lagged_set = self.create_lagged_dataset(train)
        prediction = self.predict(lagged_set)
        return prediction

    @staticmethod
    def predict(lagged_set):
        return lagged_set.iloc[-1, 0]

    def multistep_forecast(self, test_from_date, test_to_date):
        date_range = pd.date_range(test_from_date, test_to_date, freq=str(1) + "H")
        prediction = []
        for date in date_range:
            prediction = np.append(prediction,
                                   self.fit_and_predict(self.preparator, test_from_date=date))
        return self.format_prediction(prediction, self.test)

    @staticmethod
    def format_prediction(prediction, test):
        prediction = pd.Series(prediction)
        prediction.index = test[:len(prediction)].index
        prediction.index = pd.DatetimeIndex(prediction.index)
        return prediction

# model = PersistenceModel("demand_absolute", "2020-06-03", "2020-06-07")
# demand_model = PersistenceModel("demand_absolute", "2021-12-30", "2022-01-04")
# printer.print_single_forecast(model.train, model.test, model.prediction)
