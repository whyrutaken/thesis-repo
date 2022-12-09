from datetime import datetime
from pathlib2 import Path
from svr_model import SVRModel
from arima_model import ArimaModel
from lstm_model import LSTMModel
import tomli
import tomli_w
import json


def get_arima_values(config):
    p_values_ = tuple(config["arima"]["p"])
    q_values_ = tuple(config["arima"]["q"])
    d_values_ = tuple(config["arima"]["d"])
    return p_values_, q_values_, d_values_


def get_svr_values(config):
    kernel_ = tuple(config["svr"]["kernel"])
    C_ = tuple(config["svr"]["c"])
    degree_ = tuple(config["svr"]["degree"])
    coef0_ = tuple(config["svr"]["coef0"])
    gamma_ = tuple(config["svr"]["gamma"])
    return kernel_, C_, degree_, coef0_, gamma_


def get_lstm_values(config):
    dropout_ = tuple(config["lstm"]["dropout"])
    hidden_layers_ = tuple(config["lstm"]["hidden_layer"])
    activation_ = tuple(config["lstm"]["activation"])
    batch_size_ = config["lstm"]["batch_size"]
    epochs_ = config["lstm"]["epochs"]
    return dropout_, hidden_layers_, activation_, batch_size_, epochs_


def save_results(date, config, model, model_name, horizon):
    path = date + "/models/" + model_name + "-" + str(horizon) + "h"
    loss_path = path + "/loss_plots"
    Path(path).mkdir(parents=True, exist_ok=True)

    model.prediction.to_csv(path + "/predictions.csv")
    model.individual_error_scores.to_csv(path + "/individual_error_scores.csv")
    model.overall_error_scores.to_csv(path + "/overall_error_scores.csv")
    model.std_error.to_csv(path + "/std_error.csv")

    with open(path + '/best_params.txt', 'w') as fp:
        fp.write(json.dumps(model.best_params))
    with open(path + '/duration.txt', 'w') as fp:
        fp.write(str(model.duration))
    with open(date + "/config.toml", mode="wb") as fp:
        tomli_w.dump(config, fp)


# %%
if __name__ == '__main__':
    date = datetime.now().strftime("%m-%d--%H-%M")
    with open("config.toml", mode="rb") as fp:
        config = tomli.load(fp)

    attribute, test_from_date, test_to_date, horizon = config["attribute"], config["test_from_date"], config[
        "test_to_date"], config["horizon"]
    p_values, q_values, d_values = get_arima_values(config)
    kernel, C, degree, coef0, gamma = get_svr_values(config)
    dropout, hidden_layers, activation, batch_size, epochs = get_lstm_values(config)



    for i in range(1, 2):
        for h in horizon:
            arima = ArimaModel(horizon=h, attribute=attribute, test_from_date=test_from_date, test_to_date=test_to_date,
                               p_values=p_values, q_values=q_values, d_values=d_values)

            svr = SVRModel(horizon=h, attribute=attribute, test_from_date=test_from_date, test_to_date=test_to_date,
                           kernel=kernel, C=C, degree=degree, coef0=coef0, gamma=gamma)

            lstm = LSTMModel(horizon=h, attribute=attribute, test_from_date=test_from_date, test_to_date=test_to_date,
                             dropout=dropout, hidden_layers=hidden_layers, activation=activation, batch_size=batch_size,
                             epochs=epochs, file_path=date)
            save_results(date, config, svr, "SVR", h)
            save_results(date, config, lstm, "LSTM", h)
            save_results(date, config, arima, "ARIMA", h)

# %%
#  lstm = LSTMModel(attribute, test_from_date, test_to_date, horizon)
#   printer.print_single_forecast(lstm.preparator.y_train, lstm.preparator.y_test, lstm.pred)
#  printer.print_double_forecast(lstm.preparator.y_train, lstm.preparator.y_test, lstm.prediction, svr.prediction)

# printer.print_multi_forecast(svr.y_train, svr.y_test, arima.prediction, arima2.prediction, arima3.prediction)
