# %%
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error


class PersistenceModel:
    def __init__(self, attribute):
        self.master_df = pd.read_csv("extracted-data/master-df.csv")
        self.master_df = self.set_dates("2020-01-01", "2020-01-05")
        self.predictions = self.run_model(attribute)
        self.plot()

    #master_df["2020-01-01":"2020-01-07"].solar_absolute.plot()
    #plt.show()

    def set_dates(self, date1, date2):
        return self.master_df[date1:date2]

    def create_lagged_dataset(self, attribute: str) -> pd.DataFrame:
        values = pd.DataFrame(self.master_df.loc[:, attribute])
        df = pd.concat([values.shift(1), values], axis=1)
        df.columns = ['t-1', 't+1']
        print(df.head(5))
        return df

    def train_test_split(self, df):
        X = df.values
        train_size = int(len(X) * 0.66)
        train, test = X[1:train_size], X[train_size:]
        self.train_X, self.train_y = train[:, 0], train[:, 1]
        self.test_X, self.test_y = test[:, 0], test[:, 1]

    @staticmethod
    def model_persistence(x):
        return x

    def run_model(self, attribute):
        lagged_set = self.create_lagged_dataset(attribute)
        self.train_test_split(lagged_set)
        return self.predict()

    def predict(self):
        predictions = list()
        for x in self.test_X:
            yhat = self.model_persistence(x)
            predictions.append(yhat)
        test_score = mean_squared_error(self.test_y, predictions)
        print('Test MSE: %.3f' % test_score)
        return test_score

    def plot(self):
        plt.plot(self.train_y)
        plt.plot([None for i in self.train_y] + [x for x in self.test_y], color="green")
        plt.plot([None for i in self.train_y] + [x for x in self.predictions], color="red", label="prediction")
        plt.show()


persistence_model = PersistenceModel("solar_absolute")
