from datetime import datetime
from pathlib2 import Path
from svr_model import SVRModel
from arima_model import ArimaModel
from lstm_model import LSTMModel
import tomli
import tomli_w
import json


def save_results(date, config, model, model_name, horizon, iteration):
    path = date + "/models-i" + str(iteration) + "/" + model_name + "-" + str(horizon) + "h"
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
    horizon = config["horizon"]

    for i in range(1, 5):
        for h in horizon:
     #       svr = SVRModel(horizon=h, grid_search=False)
     #       save_results(date, config, svr, "SVR", h, i)
            lstm = LSTMModel(horizon=h, file_path=[date, i], grid_search=False)
            save_results(date, config, lstm, "LSTM", h, i)
     #       arima = ArimaModel(horizon=h, grid_search=False)
     #       save_results(date, config, arima, "ARIMA", h, i)

    print("End.")

# %%
#  lstm = LSTMModel(attribute, test_from_date, test_to_date, horizon)
#   printer.print_single_forecast(lstm.preparator.y_train, lstm.preparator.y_test, lstm.pred)
#  printer.print_double_forecast(lstm.preparator.y_train, lstm.preparator.y_test, lstm.prediction, svr.prediction)

# printer.print_multi_forecast(svr.y_train, svr.y_test, arima.prediction, arima2.prediction, arima3.prediction)
