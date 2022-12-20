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
        for i in range(1, 5):
            path = parent_folder + "/models-i" + str(i) + "/" + model_name + "-" + str(h) + "h"
            pred = pd.read_csv(path + "/predictions.csv", parse_dates=True, index_col=0)
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
    return best_params, cv_results, param_grid, best_score


def plot_individual_score_exp3(lstm_predictions, lstm_individual_score, lstm_std_error):
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 8)
    #   plt.errorbar(lstm_predictions[0], lstm_individual_score[0]["rmse"], yerr=lstm_std_error[0].loc["rmse"].item(),
    #                fmt=".k")
    #   plt.plot(x=lstm_predictions[0], y=lstm_individual_score[0]["rmse"])
    for i, c in zip(range(0, 4), ["red", "blue", "green"]):
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
        x_labels.append(str(h) + "h")
    fig, ax = plt.subplots()
    for i, c in zip(range(0, 4), ["red", "blue", "green", "orange"]):
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

#%%
# source: https://stackoverflow.com/questions/37161563/how-to-graph-grid-scores-from-gridsearchcv
# For plotting the results when tuning several hyperparameters, what I did was fixed all parameters to their best value
# except for one and plotted the mean score for the other parameter for each of its values.
#
def plot_grid_search_results(results, param_grid, best_params, title, model, ft_type):
    if model == "SVR":
        color = "tomato"
    else:
        color = "darkorange"

    mean_test = results['mean_test_score']
    std_test = results['std_test_score']
    mean_train = results['mean_train_score']
    std_train = results['std_train_score']

    ## Getting indices of values per hyperparameter
    masks = []
    masks_names = list(best_params.keys())
    for p_k, p_v in best_params.items():
        masks.append(list(results['param_' + p_k].data == p_v))

    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(1, len(param_grid), sharex='none', sharey='all', figsize=(20, 12))
    fig.text(s=ft_type + " forecast: Parameters of the " + model + " GridSearches during CV", x=0.5, y=0.95, fontsize=14, ha='center', va='center')
    fig.text(s=title, x=0.5, y=0.9, fontsize=12, ha='center', va='center')
  #  fig.suptitle(subtitle)
    fig.text(0.08, 0.5, 'Mean R2 score', va='center', rotation='vertical')
    for i, p in enumerate(masks_names):
        m = np.stack(masks[:i] + masks[i + 1:])
        best_parms_mask = m.all(axis=0)
        best_index = np.where(best_parms_mask)[0]
        x = np.array(param_grid[p])
        y_1 = np.array(mean_test[best_index])
        e_1 = np.array(std_test[best_index])
        y_2 = np.array(mean_train[best_index])
        e_2 = np.array(std_train[best_index])
        ax[i].errorbar(x, y_1, e_1, linestyle='--', marker='o', label='test', capsize=10, color="forestgreen")
        ax[i].errorbar(x, y_2, e_2, linestyle='-', marker='^', label='train', capsize=10, color=color)
        ax[i].set_xlabel(p.upper())
        ax[i].set_ylim(top=1.1)
        ax[i].set_ylim(bottom=-0.9)
        ax[i].grid(True, which='both')

    plt.legend()
    plt.show()



#%%
def plot_grid_search_best_scores(score1, score2, score3, horizon, model, ft_type):
    x_labels = []
    for h in horizon:
        x_labels.append(str(h) + "h")
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.text(s=ft_type + " forecast: Best scores of the " + model + " GridSearches", x=0.5, y=0.90, fontsize=14, ha='center', va='center')

  #  fig.suptitle(ft_type + " forecast: Best scores of the " + model + " GridSearches")
    ax.errorbar(x=x_labels, y=score1, color="yellowgreen", fmt='--o', label="3 months")
    ax.errorbar(x=x_labels, y=score2, color="orange", fmt='--o', label ="6 months")
    ax.errorbar(x=x_labels, y=score3, color="coral", fmt='--o', label="12 months")

    ax.set_ylabel("Best R2 score")
    ax.set_xlabel("Forecast horizon")
    ax.legend()
    ax.grid(True, which='both')
   # plt.tight_layout()
    plt.show()


def get_and_plot_results(parent_folders, model, ft_type):
    horizon = [6, 12, 24]

    results = []
    for i, ts in zip(range(len(horizon)), ["3mo", "6mo", "12mo"]):
        best_params, cv_results, param_grid, best_score = load_grid_search_results(parent_folders[i], model, horizon)
        results.append([ts, best_params, cv_results, param_grid, best_score])

        if model == "SVR":
            plot_grid_search_results(cv_results[0], param_grid[0], best_params[0],
                                     "Training set size: " + ts + ", Forecast horizon: 6h, 12h, and 24h", model, ft_type)
        else:
            for j, h in enumerate(["6h", "12h", "24h"]):
                plot_grid_search_results(cv_results[j], param_grid[j], best_params[j],
                                     "Training set size: " + ts + ", Forecast horizon: " + h, model, ft_type)
    plot_grid_search_best_scores(results[0][-1], results[1][-1], results[2][-1], horizon, model, ft_type)
    return results


#%%
svr = "SVR"
lstm = "LSTM"
pv = "PV"
cons = "CP"

parent_folders = ["12-17--23-00-pv-svr-3mo-grid", "12-17--22-39-pv-svr-6mo-grid", "12-18--00-33-pv-svr-1y-grid"]
svr_pv = get_and_plot_results(parent_folders, svr, pv)
#%%
# TODO: 1-year Cons forecast!!!
parent_folders = ["12-20--04-10-cf-svr-3mo-grid", "12-18--03-45-cf-svr-6mo-grid", "12-18--00-33-pv-svr-1y-grid"]
svr_cons = get_and_plot_results(parent_folders, svr, cons)

#%%
parent_folders = ["12-19--20-40-pv-lstm1-3mo-grid", "12-19--20-41-pv-lstm1-6mo-grid", "12-20--03-38-pv-lstm1-12-mo-grid"]
lstm1_pv = get_and_plot_results(parent_folders, lstm, pv)

#%%
parent_folders = ["12-20--04-24-cf-lstm1-3mo-grid", "12-20--04-22-cf-lstm1-6mo-grid", "12-20--16-49-cf-lstm1-1y-grid"]
lstm1_cons = get_and_plot_results(parent_folders, lstm, cons)



#%%
#%%



# %%
parent_folder = "12-16--06-35-arima-winter"
# lstm_predictions, lstm_individual_error, lstm_overall_error, lstm_std_error = load_training_data(parent_folder, "LSTM", horizon)
arima_predictions, arima_individual_error, arima_overall_error, arima_std_error = load_training_data(parent_folder,
                                                                                                     "ARIMA", horizon)
# svr_predictions, svr_individual_error, svr_overall_error, svr_std_error = load_training_data(parent_folder, "SVR", horizon)
# %%
plot_individual_score_exp3(arima_predictions, arima_individual_error, arima_std_error)
plot_overall_score_exp1(arima_overall_error, arima_std_error, horizon)
