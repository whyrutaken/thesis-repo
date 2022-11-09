# %%
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error


class PersistenceModel:
    def __init__(self, attribute, from_date, to_date):
        self.master_df = pd.read_csv("extracted-data/master-df.csv")
        self.df = self.set_attribute_and_dates(attribute, from_date, to_date)
        #     self.plot_initial(attribute)
        self.predictions = self.run_and_predict_model()
        self.test_score = self.test_score()
        self.plot()

    def plot_initial(self, attribute):
        self.master_df.loc[:, attribute].plot()
        plt.show()

    def set_attribute_and_dates(self, attr, date1, date2, ):
        return self.master_df[date1:date2].loc[:, attr]

    def create_lagged_dataset(self) -> pd.DataFrame:
        values = pd.DataFrame(self.df)
        df = pd.concat([values.shift(1), values], axis=1)
        df.columns = ['t-1', 't+1']
        print(df.head(5))
        return df

    def train_test_split(self, df):
        X = df.values
        train_size = int(len(X) * 0.6)
        train, test = X[1:train_size], X[train_size:]
        self.train_X, self.train_y = train[:, 0], train[:, 1]
        self.test_X, self.test_y = test[:, 0], test[:, 1]

    @staticmethod
    def model_persistence(x):
        return x

    def run_and_predict_model(self):
        lagged_set = self.create_lagged_dataset()
        self.train_test_split(lagged_set)
        return self.predict()

    def predict(self):
        predictions = list()
        for x in self.test_X:
            yhat = self.model_persistence(x)
            predictions.append(yhat)
        return predictions

    def test_score(self):
        test_score = mean_squared_error(self.test_y, self.predictions)
        print('Test MSE: %.3f' % test_score)
        return test_score

    def plot(self):
        plt.plot(self.train_y)
        plt.plot([None for i in self.train_y] + [x for x in self.test_y], color="green")
        plt.plot([None for i in self.train_y] + [x for x in self.predictions], color="red", label="prediction")
        plt.show()


solar_model = PersistenceModel("solar_absolute", "2020-06-08", "2020-06-12")
demand_model = PersistenceModel("demand_absolute", "2020-06-08", "2020-06-12")