from datetime import datetime
import pandas as pd
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense
import numpy as np
from keras.wrappers.scikit_learn import KerasRegressor
from pathlib2 import Path
from sklearn.model_selection import GridSearchCV
from preparator import Preparator
import matplotlib.pyplot as plt
from error_metric_calculator import Metrics
import tomli


class LSTMModel:
    def __init__(self, horizon: int, file_path: list, grid_search: bool):
        # hyperparameters
        attribute, test_from_date, test_to_date, dropout, hidden_layers, activation, self.batch_size, self.epochs, best_params = self.read_config()
        self.preparator = Preparator(attribute, test_from_date)

        if grid_search:
            self.best_params = self.grid_search(test_from_date, horizon, self.lstm1, dropout, hidden_layers, activation)
        else:
            self.best_params = best_params
            self.prediction, self.duration = self.multistep_forecast(test_from_date, test_to_date, horizon, file_path,
                                                                     best_params)

            self.individual_error_scores, self.overall_error_scores = Metrics().calculate_errors(
                self.preparator.y_test[horizon:], self.prediction)
            self.std_error = self.individual_error_scores.std()

    #     printer.print_single_forecast(self.preparator.y_train, self.preparator.y_test, self.prediction)

    @staticmethod
    def read_config():
        with open("config.toml", mode="rb") as fp:
            config = tomli.load(fp)
        attribute_, test_from_date_, test_to_date_ = config["attribute"], config["test_from_date"], config[
            "test_to_date"]
        dropout_ = tuple(config["lstm"]["dropout"])
        hidden_layers_ = tuple(config["lstm"]["hidden_layer"])
        activation_ = tuple(config["lstm"]["activation"])
        batch_size_ = config["lstm"]["batch_size"]
        epochs_ = config["lstm"]["epochs"]
        best_params_ = config["lstm"]["best_params"]
        return attribute_, test_from_date_, test_to_date_, dropout_, hidden_layers_, activation_, batch_size_, epochs_, best_params_

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

    def split_data(self, test_from_date, horizon):
        x_train, x_test, y_train, y_test = self.preparator.get_scaled_data(test_from_date)
        x_train, y_train = self.prepare_sliding_windows(x_train, y_train, horizon)
        x_test, y_test = self.prepare_sliding_windows(x_test, y_test, horizon)
        x_test, y_test = x_test[:horizon, :], y_test[:horizon]
        return x_train, x_test, y_train, y_test

    @staticmethod
    def lstm1(hidden_layer, dropout, input_shape, activation):
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
        model = Sequential()
        model.add(LSTM(64, input_shape=input_shape, activation=activation,
                       return_sequences=True, dropout=dropout))
        model.add(LSTM(hidden_layer, activation=activation, return_sequences=False, dropout=dropout))
        model.add(Dense(32))
        model.add(Dense(1))
        model.compile(loss='mse', optimizer='adam')
        model.summary()
        return model

    @staticmethod
    def set_callbacks():
        early_stopper = EarlyStopping(monitor='loss', patience=20, mode="min", restore_best_weights=True, verbose=2)
        return early_stopper

    def grid_search(self, test_from_date, horizon, build_model, dropout, hidden_layers, activation):
        x_train, x_test, y_train, y_test = self.split_data(test_from_date, horizon)
        start = datetime.now()
        print("LSTM Grid search started")
        input_shape = (x_train.shape[1], x_train.shape[2])
        model = KerasRegressor(
            build_fn=build_model,
            hidden_layer=hidden_layers, dropout=dropout, input_shape=input_shape,
            activation=activation
        )
        hyperparameters = dict(hidden_layer=hidden_layers, dropout=dropout, activation=activation)

        # get rid of the cross-validation
        # source: https://stackoverflow.com/questions/44636370/scikit-learn-gridsearchcv-without-cross-validation-unsupervised-learning
        cv = [(slice(None), slice(None))]

        rs = GridSearchCV(model, param_grid=hyperparameters, cv=cv, n_jobs=-1)
        rs.fit(x_train, y_train, verbose=1)

        self.print_end(start, "Total duration of LSTM grid search: ")
        return rs.best_params_

    def fit_and_predict(self, test_from_date, horizon, best_params, file_path):
        start = datetime.now()
        print("LSTM fit and predict, test_from_date={}, horizon={}, best_params={}".format(test_from_date, horizon,
                                                                                           best_params))
        x_train, x_test, y_train, y_test = self.split_data(test_from_date, horizon)
        input_shape = (x_train.shape[1], x_train.shape[2])
        my_callbacks = self.set_callbacks()
        model = self.lstm1(best_params["hidden_layer"], best_params["dropout"], input_shape, best_params["activation"])
        history = model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, callbacks=my_callbacks,
                            verbose=1, shuffle=False, validation_data=(x_test, y_test))
        self.plot_loss(history, file_path, horizon)
        predictions = []
        for x in x_test:
            pred = model.predict(np.reshape(x, (1, x_test.shape[1], x_test.shape[2])))
            inverse_scaled_pred = self.preparator.inverse_scaler(pred)
            predictions.append(inverse_scaled_pred.ravel())

        self.print_end(start, "Duration: ")
        return predictions

    def multistep_forecast(self, test_from_date, test_to_date, horizon, file_path, best_params):
        start = datetime.now()
        date_range = pd.date_range(test_from_date, test_to_date, freq=str(horizon) + "H")
        predictions = []
        for date in date_range:
            predictions = np.append(predictions, self.fit_and_predict(date, horizon, best_params, file_path))
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
    def plot_loss(history, file_path, horizon):
        file_path = file_path[0] + "/models-i" + str(file_path[1]) + "/LSTM-" + str(horizon) + "h/loss_plots"
        Path(file_path).mkdir(parents=True, exist_ok=True)
        time = datetime.now().strftime("%H-%M-%S")
        plt.figure()
        plt.plot(history.history['loss'], label='training loss')
        plt.plot(history.history['val_loss'], label='validation loss')
        plt.title('LSTM training vs validation loss')
        plt.ylabel('MSE')
        plt.xlabel('Epochs')
        plt.legend()
        plt.savefig(file_path + "/loss_" + time + ".png")
#  plt.show()

# lstm = LSTMModel("solar_absolute", test_from_date="2020-01-10 00:00", test_to_date="2020-01-11 00:00", horizon=12)
