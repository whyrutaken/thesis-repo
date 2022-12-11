import tomli
import tomli_w
from pathlib2 import Path
from datetime import datetime
from svr_model import SVRModel
from arima_model import ArimaModel
from lstm_model import LSTMModel
import json


def save_results(date, horizon, model_name, best_params, config):
    path = date + "/grid_search/" + model_name + "-" + str(horizon) + "h"
    Path(path).mkdir(parents=True, exist_ok=True)
    with open(path + '/best_params.txt', 'w') as fp:
        fp.write(json.dumps(best_params))
    with open(date + "/config.toml", mode="wb") as fp:
        tomli_w.dump(config, fp)


if __name__ == '__main__':
    date = datetime.now().strftime("%m-%d--%H-%M")
    with open("config.toml", mode="rb") as fp:
        config = tomli.load(fp)
    horizon = config["horizon"]

    for h in horizon:
        lstm_gs = LSTMModel(horizon=h, file_path=[""], grid_search=True)
        save_results(date, h, "LSTM", lstm_gs.best_params, config)
        arima_gs = ArimaModel(horizon=h, grid_search=True)
        save_results(date, h, "ARIMA", arima_gs.best_params, config)
   #     svr_gs = SVRModel(horizon=h, grid_search=True)
   #     save_results(date, h, "SVR", svr_gs.best_params, config)
