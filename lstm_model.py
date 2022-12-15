from datetime import datetime
import pandas as pd
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense
import numpy as np
from keras.wrappers.scikit_learn import KerasRegressor
from pathlib2 import Path
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from preparator import Preparator
import matplotlib.pyplot as plt
from error_metric_calculator import Metrics
import tomli
import tensorflow as tf


class LSTMModel:
    def __init__(self, horizon: int, file_path: list, grid_search: bool):
        # hyperparameters
        attribute, train_from_date, test_from_date, test_to_date, dropout, hidden_layers, activation, batch_size, epochs, best_params = self.read_config()
        self.preparator = Preparator(attribute, train_from_date=train_from_date, test_from_date=test_from_date)

        if grid_search:
            self.best_params, self.cv_results, self.param_grid, self.best_score = self.grid_search(train_from_date=train_from_date,
                                                                                  test_from_date=test_from_date,
                                                                                  horizon=horizon,
                                                                                  build_model=self.lstm2,
                                                                                  dropout=dropout,
                                                                                  hidden_layers=hidden_layers,
                                                                                  activation=activation, epochs=epochs,
                                                                                  batch_size=batch_size)
        else:
            self.best_params = best_params
            self.prediction, self.duration = self.multistep_forecast(train_from_date=train_from_date,
                                                                     test_from_date=test_from_date,
                                                                     test_to_date=test_to_date, horizon=horizon,
                                                                     file_path=file_path,
                                                                     best_params=best_params)

            self.individual_error_scores, self.overall_error_scores = Metrics().calculate_errors(
                self.preparator.y_test[horizon:], self.prediction)
            self.std_error = self.individual_error_scores.std()

    #     printer.print_single_forecast(self.preparator.y_train, self.preparator.y_test, self.prediction)

    @staticmethod
    def read_config():
        with open("config.toml", mode="rb") as fp:
            config = tomli.load(fp)
        attribute_, train_from_date_, test_from_date_, test_to_date_ = config["attribute"], config["train_from_date"], config["test_from_date"], config["test_to_date"]
        dropout_ = tuple(config["lstm"]["dropout"])
        hidden_layers_ = tuple(config["lstm"]["hidden_layer"])
        activation_ = tuple(config["lstm"]["activation"])
        batch_size_ = config["lstm"]["batch_size"]
        epochs_ = config["lstm"]["epochs"]
        best_params_ = config["lstm"]["best_params"]
        return attribute_, train_from_date_, test_from_date_, test_to_date_, dropout_, hidden_layers_, activation_, batch_size_, epochs_, best_params_

    @staticmethod
    def prepare_sliding_windows(feature, target, sliding_window):
        X, Y = [], []
        for i in range(len(feature) - sliding_window):
            X.append(feature[i:(i + sliding_window), :])  # features
            Y.append(target[i + sliding_window, -1])  # target value (weighted_price)
        X = np.array(X)
        Y = np.array(Y)
        Y = Y.reshape(Y.shape[0], 1)
        return X, Y

    def split_data(self, train_from_date, test_from_date, horizon):
        x_train, x_test, y_train, y_test = self.preparator.get_scaled_data(test_from_date=test_from_date,
                                                                           train_from_date=train_from_date)
        x_train, y_train = self.prepare_sliding_windows(x_train, y_train, horizon)
        x_test, y_test = self.prepare_sliding_windows(x_test, y_test, horizon)
        x_test, y_test = x_test[:horizon, :], y_test[:horizon]
        return x_train, x_test, y_train, y_test

    @staticmethod
    def lstm1(hidden_layer, dropout, input_shape, activation):
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            model = Sequential()
            model.add(LSTM(hidden_layer, input_shape=input_shape, activation=activation,
                           return_sequences=False, dropout=dropout))
            model.add(Dense(hidden_layer))
            model.add(Dense(1))
            model.compile(loss='mse', optimizer='adam')
        model.summary()
        return model

    @staticmethod
    def lstm2(hidden_layer, dropout, input_shape, activation):
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            model = Sequential()
            model.add(LSTM(64, input_shape=input_shape, activation=activation,
                           return_sequences=True))
            model.add(LSTM(hidden_layer, activation=activation, return_sequences=False, dropout=dropout))
            model.add(Dense(32))
            model.add(Dense(1))
            model.compile(loss='mse', optimizer='adam')
        model.summary()
        return model

    @staticmethod
    def set_callbacks():
        early_stopper = EarlyStopping(monitor='loss', patience=10, mode="min", restore_best_weights=True, verbose=2)
        return early_stopper

    def grid_search(self, train_from_date, test_from_date, horizon, build_model, dropout, hidden_layers, activation,
                    epochs, batch_size):
        start = datetime.now()
        x_train, x_test, y_train, y_test = self.split_data(test_from_date=test_from_date,
                                                           train_from_date=train_from_date, horizon=horizon)
        print("LSTM Grid search started")
        input_shape = (x_train.shape[1], x_train.shape[2])
        my_callbacks = self.set_callbacks()
        model = KerasRegressor(
            build_fn=build_model,
            hidden_layer=hidden_layers, input_shape=input_shape,
            epochs=epochs, batch_size=batch_size, verbose=2, callbacks=my_callbacks
        )
        hyperparameters = dict(hidden_layer=hidden_layers, batch_size=batch_size, epochs=epochs, dropout=dropout, activation=activation)

        cv = TimeSeriesSplit()
        rs = GridSearchCV(model, param_grid=hyperparameters, verbose=3, return_train_score=True, cv=cv) # n_jobs=-1,
        rs.fit(x_train, y_train)

        self.print_end(start, "Total duration of LSTM grid search: ")
        return rs.best_params_, rs.cv_results_, rs.param_grid, rs.best_score_

    def fit_and_predict(self, train_from_date, test_from_date, horizon, best_params, file_path):
        start = datetime.now()
        print("LSTM fit and predict, train_from_date={}, test_from_date={}, horizon={}, best_params={}".format(
            train_from_date, test_from_date, horizon, best_params))
        x_train, x_test, y_train, y_test = self.split_data(test_from_date=test_from_date,
                                                           train_from_date=train_from_date, horizon=horizon)
        input_shape = (x_train.shape[1], x_train.shape[2])
        my_callbacks = self.set_callbacks()
        model = self.lstm1(best_params["hidden_layer"], best_params["dropout"], input_shape, best_params["activation"])

        history = model.fit(x_train, y_train, epochs=best_params["epochs"], batch_size=best_params["batch_size"],
                            callbacks=my_callbacks,
                            verbose=1, shuffle=False, validation_data=(x_test, y_test))
        self.plot_loss(history, file_path, horizon)
        self.save_model(model, file_path, horizon)
        predictions = []
        for x in x_test:
            pred = model.predict(np.reshape(x, (1, x_test.shape[1], x_test.shape[2])))
            inverse_scaled_pred = self.preparator.inverse_scaler(pred)
            predictions.append(inverse_scaled_pred.ravel())

        self.print_end(start, "Duration: ")
        return predictions

    def multistep_forecast(self, train_from_date, test_from_date, test_to_date, horizon, file_path, best_params):
        start = datetime.now()
        date_range = pd.date_range(test_from_date, test_to_date, freq=str(horizon) + "H")
        predictions = []
        for date in date_range:
            predictions = np.append(predictions,
                                    self.fit_and_predict(train_from_date=train_from_date, test_from_date=date,
                                                         horizon=horizon, best_params=best_params, file_path=file_path))
        duration = self.print_end(start, "Total duration of LSTM multistep forecast: ")
        return self.format_prediction(predictions, horizon), duration

    @staticmethod
    def print_end(start, note):
        end = datetime.now()
        duration = end - start
        print(note, duration)
        return duration

    def format_prediction(self, prediction, horizon):
        prediction = pd.Series(prediction)
        prediction.index = self.preparator.y_test[horizon:len(prediction) + horizon].index
        prediction.index = pd.DatetimeIndex(prediction.index)
        return prediction

    @staticmethod
    def save_model(model, file_path, horizon):
        time = datetime.now().strftime("%H%M%S")
        file_path = file_path[0] + "/models-i" + str(file_path[1]) + "/LSTM-" + str(horizon) + "h/model"
        model.save(file_path + "/model" + str(time) + ".h5")

    @staticmethod
    def plot_loss(history, file_path, horizon):
        file_path = file_path[0] + "/models-i" + str(file_path[1]) + "/LSTM-" + str(horizon) + "h/loss_plots"
        Path(file_path).mkdir(parents=True, exist_ok=True)
        time = datetime.now().strftime("%H%M%S")
        plt.figure()
        plt.plot(history.history['loss'], label='training loss')
        plt.plot(history.history['val_loss'], label='validation loss')
        plt.title('LSTM training vs validation loss')
        plt.ylabel('MSE')
        plt.xlabel('Epochs')
        plt.legend()
        plt.savefig(file_path + "/loss_" + time + ".png")
        #  plt.show()
