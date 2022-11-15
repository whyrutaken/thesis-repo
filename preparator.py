import pandas as pd


class Preparator:
    def __init__(self, attribute):
        self.master_df = pd.read_csv("extracted-data/master-df.csv")
        self.df = self.set_attribute_and_dates(attribute)

    def set_attribute_and_dates(self, attribute, from_date="2020-01-01", to_date="2022-02-28") -> pd.DataFrame:
        return self.master_df[from_date:to_date].loc[:, attribute]

    def train_test_split(self, train_from_date="2020-01-01", test_from_date="2022-01-01"):
        train = pd.Series(self.df[train_from_date:test_from_date], index=self.df[train_from_date:test_from_date].index)
        test = pd.Series(self.df[test_from_date:], index=self.df[test_from_date:].index)
        train.index = pd.DatetimeIndex(train.index)
        test.index = pd.DatetimeIndex(test.index)
        return train, test

