import pandas as pd
from keras import Sequential
from keras.callbacks import EarlyStopping, CSVLogger
from keras.layers import LSTM, Dense
import numpy as np
from preparator import Preparator
import matplotlib.pyplot as plt
from error_metric_calculator import Metrics
import printer

class LSTMModel:
    def __init__(self, attribute, test_from_date, test_to_date, horizon):


        batch_size = 168
        epochs = 10



        self.preparator = Preparator(attribute, test_from_date)
        self.x_train, self.x_test, self.y_train, self.y_test = self.preparator.get_scaled_data(test_from_date)

        self.pred = self.fit_and_predict(test_from_date, self.x_train, self.y_train, self.x_test, self.y_test, horizon, batch_size, epochs)
        self.pred = self.format_prediction(self.pred, horizon)
        self.individual_scores, self.overall_scores = Metrics().calculate_errors(self.preparator.y_test, self.pred[horizon:])
        printer.print_single_forecast(self.preparator.y_train, self.preparator.y_test, self.pred)

    #    self.prediction = self.multistep_forecast(test_from_date=test_from_date, test_to_date=test_to_date, x_train=self.x_train, y_train=self.y_train, x_test=self.x_test, y_test=self.y_test, horizon=horizon)

    def prepare_sliding_windows(self, feature, target, sliding_window):
        X, Y = [], []
        for i in range(len(feature) - sliding_window):
            X.append(feature[i:(i + sliding_window), :])  # features
            Y.append(target[i + sliding_window, -1])  # target value (weighted_price)
        X = np.array(X)
        Y = np.array(Y)
        Y = Y.reshape(Y.shape[0],1)
        return X, Y


    def build_model(self, n_timesteps, n_features):
        model = Sequential()
        model.add(LSTM(64, input_shape=(n_timesteps, n_features), activation="relu",
                       return_sequences=True))
        model.add(LSTM(32, activation='relu', return_sequences=False))
        model.add(Dense(16))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')
        model.summary()
        return model

    def fit_and_predict(self, test_from_date, x_train, y_train, x_test, y_test, horizon, batch_size, epochs):
        x_train, y_train = self.prepare_sliding_windows(x_train, y_train, horizon)
        x_test, y_test = self.prepare_sliding_windows(x_test, y_test, horizon)

        model = self.build_model(x_train.shape[1], x_train.shape[2])
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=False, validation_split=0.1, validation_batch_size=batch_size)
        self.plot(history)
        prediction = model.predict(x_test)
        inverse_scaled_prediction = self.preparator.inverse_scaler(prediction)
        return inverse_scaled_prediction.ravel()

    def multistep_forecast(self, test_from_date, test_to_date, x_train, y_train, x_test, y_test, horizon):
        date_range = pd.date_range(test_from_date, test_to_date, freq=str(horizon) + "H")
        prediction = []
        for date in date_range:
            prediction = np.append(prediction, self.fit_and_predict(date, x_train, y_train, x_test, y_test, horizon))
        return self.format_prediction(prediction, horizon)

    def format_prediction(self, prediction, horizon):
        prediction = pd.Series(prediction)
        prediction.index = self.preparator.y_test[horizon:len(prediction)+horizon].index
        prediction.index = pd.DatetimeIndex(prediction.index)
        prediction = prediction[:-horizon]
        return prediction

    def plot(self, history):
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()


#lstm = LSTMModel("demand_absolute", test_from_date="2021-06-10 00:00", test_to_date="2021-06-24 00:00", horizon=168)

