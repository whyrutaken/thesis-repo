import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import deepdish as dd
import tomli

def load_training_data(parent_folder, model_name, horizon):


    predictions = []
    individual_error_scores = []
    overall_error_scores = []
    std_error = []
    for h in horizon:
        for i in range(1, 2):
            path = parent_folder + "/models-i" + str(i) + "/" + model_name + "-" + str(h) + "h"
            pred = pd.read_csv("/predictions.csv", parse_dates=True, index_col=0)
            pred.index = pd.DatetimeIndex(pred.index)
            predictions.append(pred)
            ind_error = pd.read_csv(path + "/individual_error_scores.csv", parse_dates=True, index_col=0)
            ind_error.index = pd.DatetimeIndex(ind_error.index)
            individual_error_scores.append(ind_error)
            overall_error_scores.append(pd.read_csv(path + "/overall_error_scores.csv", index_col=0))
            std_error.append(
                pd.read_csv(path + "/std_error.csv", index_col=0))
    return predictions, individual_error_scores, overall_error_scores, std_error


def load_grid_search_results(parent_folder, model_name, horizon):
    with open(parent_folder + "/config.toml", mode="rb") as fp:
        config = tomli.load(fp)
 #   horizon = config["horizon"]
    best_params = []
    cv_results = []
    param_grid = []
    best_score = []
    for h in horizon:
        path = parent_folder + "/grid_search/" + model_name + "-" + str(h) + "h"
        best_params.append(dd.io.load(path + "/best_params.h5"))
        cv_results.append(dd.io.load(path + "/cv_results.h5"))
        param_grid.append(dd.io.load(path + "/param_grid.h5"))
        best_score.append(dd.io.load(path + "/best_score.h5"))
    return best_params, cv_results, param_grid, best_score, horizon


def plot_individual_score_exp3(lstm_predictions, lstm_individual_score, lstm_std_error):
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 8)
    #   plt.errorbar(lstm_predictions[0], lstm_individual_score[0]["rmse"], yerr=lstm_std_error[0].loc["rmse"].item(),
    #                fmt=".k")
    #   plt.plot(x=lstm_predictions[0], y=lstm_individual_score[0]["rmse"])
    for i, c in zip(range(0, 1), ["red", "blue", "green"]):
        error = lstm_std_error[i].loc["rmse"].item()
        #  lstm_predictions[i].plot(use_index=True, ax=ax)
        #    lstm_individual_score[i]["rmse"].plot(use_index=True, ax=ax, color=c)
        ax.errorbar(x=lstm_individual_score[i]["rmse"].index, y=lstm_individual_score[i]["rmse"], yerr=error,
                    capsize=10,
                    errorevery=(3 * i, 22), color=c, fmt='-')
    #    ax.errorbar(x=lstm_predictions[i].index, y=lstm_predictions[i].iloc[:,0], yerr=error, errorevery=(3 * i, 22), color=c, fmt='-')

    ax.legend(["error1", "error2", "error3"])
    ax.set_ylabel("RMSE [Wh]")
    ax.set_xlabel("Time")
    ax.grid(True, which='both')
    plt.show()


def plot_overall_score_exp1(scores, std_error, horizon):
    x_labels = []
    for h in horizon:
        x_labels.append(str(h)+"h")
    fig, ax = plt.subplots()
    for i, c in zip(range(0, 1), ["red", "blue", "green", "orange", "magenta"]):
        error = std_error[i].loc["rmse"].item()
        #  lstm_predictions[i].plot(use_index=True, ax=ax)
        #    lstm_individual_score[i]["rmse"].plot(use_index=True, ax=ax, color=c)
        ax.errorbar(x=x_labels, y=scores[i]["rmse"], yerr=error, color=c, fmt='o', capsize=10)
    #    ax.errorbar(x=lstm_predictions[i].index, y=lstm_predictions[i].iloc[:,0], yerr=error, errorevery=(3 * i, 22), color=c, fmt='-')

    ax.legend(["error1", "error2", "error3"])
    ax.set_ylabel("R-squared")
    ax.set_xlabel("Time")
    ax.grid(True, which='both')
    plt.show()



# source: https://stackoverflow.com/questions/37161563/how-to-graph-grid-scores-from-gridsearchcv
# For plotting the results when tuning several hyperparameters, what I did was fixed all parameters to their best value
# except for one and plotted the mean score for the other parameter for each of its values.
#
def plot_grid_search_results(results, param_grid, best_params):
    mean_test = results['mean_test_score']
    std_test = results['std_test_score']
    mean_train = results['mean_train_score']
    std_train = results['std_train_score']

    ## Getting indices of values per hyperparameter
    masks = []
    masks_names = list(best_params.keys())
    for p_k, p_v in best_params.items():
        masks.append(list(results['param_' + p_k].data == p_v))

    fig, ax = plt.subplots(1, len(param_grid), sharex='none', sharey='all', figsize=(25,10))
    fig.suptitle('Score per parameter')
    fig.text(0.1, 0.5, 'MEAN SCORE', va='center', rotation='vertical')
    for i, p in enumerate(masks_names):
        m = np.stack(masks[:i] + masks[i + 1:])
        best_parms_mask = m.all(axis=0)
        best_index = np.where(best_parms_mask)[0]
        x = np.array(param_grid[p])
        y_1 = np.array(mean_test[best_index])
        e_1 = np.array(std_test[best_index])
        y_2 = np.array(mean_train[best_index])
        e_2 = np.array(std_train[best_index])
        ax[i].errorbar(x, y_1, e_1, linestyle='--', marker='o', label='test', capsize=10)
        ax[i].errorbar(x, y_2, e_2, linestyle='-', marker='^', label='train', capsize=10)
        ax[i].set_xlabel(p.upper())
        ax[i].set_ylim(top=1)
        ax[i].set_ylim(bottom=0)
   #     ax[i].set_ylim(bottom=min(np.amin(mean_train), np.amin(mean_test)))
        ax[i].grid(True, which='both')

    plt.legend()
    plt.show()

def plot_grid_search_best_scores(scores, horizon):
    x_labels = []
    for h in horizon:
        x_labels.append(str(h)+"h")
    fig, ax = plt.subplots()
    ax.errorbar(x=x_labels, y=scores, color="red", fmt='--o')

#    ax.legend(["error1", "error2", "error3"])
    ax.set_ylabel("R-squared")
    ax.set_xlabel("Forecast horizon")
    ax.grid(True, which='both')
    plt.show()



parent_folder = "12-15--00-12_svr_grid"
horizon = [6]
svr = "SVR"
lstm = "LSTM"

svr_best_params, svr_cv_results, svr_param_grid, svr_best_score, svr_horizon = load_grid_search_results(parent_folder, svr, horizon)
plot_grid_search_results(svr_cv_results[0], svr_param_grid[0], svr_best_params[0])
plot_grid_search_best_scores(svr_best_score, horizon)

parent_folder = "12-15--15-39"
svr2_bp, svr2_cv_res, svr2_pg, svr2_bs, svr2_hor = load_grid_search_results(parent_folder, svr, horizon)
plot_grid_search_results(svr2_cv_res[0], svr2_pg[0], svr2_bp[0])
plot_grid_search_best_scores(svr2_bs, horizon)

# %%
lstm_predictions, lstm_individual_error, lstm_overall_error, lstm_std_error = load_training_data(parent_folder, "LSTM",
                                                                                                 horizon)
arima_predictions, arima_individual_error, arima_overall_error, arima_std_error = load_training_data(parent_folder,
                                                                                                     "ARIMA",
                                                                                                     horizon)
svr_predictions, svr_individual_error, svr_overall_error, svr_std_error = load_training_data(parent_folder, "SVR",
                                                                                             horizon)

# plot_individual_score_exp3(lstm_predictions, lstm_individual_error, lstm_std_error)
plot_overall_score_exp1(lstm_overall_error, lstm_std_error)
