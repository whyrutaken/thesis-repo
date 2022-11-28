import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from preparator import Preparator
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import numpy as np
import datetime

def print_forecast(train, test, prediction):
    plt.figure(figsize=(15, 5), dpi=100)
    plt.locator_params(axis='x', nbins=5)
    plt.plot(train[len(train) - 48:], label='training')
    plt.plot(test[:168], label='actual')
    plt.plot(prediction, label='forecast')

    plt.title('Forecast vs Actual')
    plt.legend(loc='upper right', fontsize=8)
    plt.xlabel('Time')
    plt.ylabel('PV production [Wh]')
    plt.show()


class SVRModel:
    def __init__(self, attribute):


        self.preparator = Preparator(attribute)
     #   self.prediction = self.fit_and_predict(test_from_date="2022-01-01 00:00", horizon=24)
     #   self.prediction = self.multistep_forecast(test_from_date="2021-01-01 02:00")
        self.pred = self.fit_and_predict_all()

    def fit_and_predict(self, test_from_date, horizon=1):
        x_train, x_test, y_train, y_test = self.preparator.get_scaled_data(test_from_date)
        model = SVR()
        model = model.fit(x_train[:-2], y_train.ravel())
        test = x_test[0].reshape(1, 9)
        prediction = model.predict(test)
        inverse_scaled_prediction = self.preparator.inverse_scaler(prediction)
        return self.format_prediction(inverse_scaled_prediction[0][0])

    def fit_and_predict_all(self):
        x_train, x_test, y_train, y_test = self.preparator.get_scaled_data(test_from_date="2022-01-01 00:00")
        model = SVR(kernel="rbf")
        model = model.fit(x_train[:-2], y_train.ravel())
       # test = x_test[test_index].reshape(1, 6)
        prediction = model.predict(x_test[:168])
        inv_pred = list()
        for pred in prediction:
            inv_pred.append(self.preparator.inverse_scaler(pred)[0])
        return self.format_prediction(inv_pred)

    def format_prediction(self, prediction):
        prediction = pd.Series(prediction)
        prediction.index = self.preparator.y_test[:len(prediction)].index
        prediction.index = pd.DatetimeIndex(prediction.index)
        return prediction

    def multistep_forecast(self, test_from_date):
        prediction = list()
        for size in train_size:
            prediction.append(
                self.fit_and_predict(size))
        return self.format_prediction(prediction)


model = SVRModel("solar_absolute")

# %%


# %%


# scaled_prediction = format_prediction(scaled_prediction[0][:1200], preparator.y_test)

print_forecast(model.preparator.y_train, model.preparator.y_test, model.pred)
# print(model.model.score(x_test, y_test))
