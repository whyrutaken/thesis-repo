import tomli
import tomli_w
from pathlib2 import Path
from datetime import datetime
from svr_model import SVRModel
from arima_model import ArimaModel
from lstm_model import LSTMModel
import deepdish as dd


def save_results(date, horizon, model_name, best_params, cv_results, param_grid, best_score, config):
    path = date + "/grid_search/" + model_name + "-" + str(horizon) + "h"
    Path(path).mkdir(parents=True, exist_ok=True)

    dd.io.save(path + "/cv_results.h5", cv_results)
    dd.io.save(path + '/best_params.h5', best_params)
    dd.io.save(path + "/param_grid.h5", param_grid)
    dd.io.save(path + "/best_score.h5", best_score)

    if horizon == 6:
        with open(date + "/config.toml", mode="wb") as fp:
            tomli_w.dump(config, fp)


if __name__ == '__main__':
    date = datetime.now().strftime("%m-%d--%H-%M")
    with open("config.toml", mode="rb") as fp:
        config = tomli.load(fp)
    horizon = config["horizon"]

    #   svr_gs = SVRModel(horizon=horizon[-1], grid_search=True)
    #   save_results(date, horizon[-1], "SVR", svr_gs.best_params, svr_gs.cv_results, svr_gs.param_grid, svr_gs.best_score, config)

   # lstm_gs = LSTMModel(config, horizon=horizon[-1], file_path=[""], grid_search=True)
   # save_results(date, horizon[-1], "LSTM", lstm_gs.best_params, lstm_gs.cv_results,lstm_gs.param_grid, lstm_gs.best_score, config)

   # for h in horizon:
    h=6
   #     lstm_gs = LSTMModel(config, horizon=h, file_path=[""], grid_search=True)
    #    save_results(date, h, "LSTM", lstm_gs.best_params, lstm_gs.cv_results, lstm_gs.param_grid, lstm_gs.best_score, config)
   #     arima_gs = ArimaModel(config, horizon=h, grid_search=True)
   #     save_results(date, h, "ARIMA", arima_gs.best_params, "", config)
  #  svr_gs = SVRModel(config, horizon=h, file_path=[""], grid_search=True)
  #  save_results(date, h, "SVR", svr_gs.best_params, svr_gs.cv_results, svr_gs.param_grid, svr_gs.best_score, config)

    print("End.")
