#####
#   Functions used for loading results, plotting different graphs, etc.
#

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import deepdish as dd


### Load results

def load_model_results_with_iteration(parent_folder, model_name, horizon):
    iteration = []
    for i in range(1, 5):
        predictions = []
        individual_error_scores = []
        overall_error_scores = []
        std_error = []
        for h in horizon:
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
        iteration.append([predictions, individual_error_scores, overall_error_scores, std_error])
    return iteration


def load_model_results_without_iteration(parent_folder, model_name, horizon):
    path = parent_folder + "/models-i/" + model_name + "-" + str(horizon) + "h"
    pred = pd.read_csv(path + "/predictions.csv", parse_dates=True, index_col=0)
    pred.index = pd.DatetimeIndex(pred.index)
    ind_error = pd.read_csv(path + "/individual_error_scores.csv", parse_dates=True, index_col=0)
    ind_error.index = pd.DatetimeIndex(ind_error.index)
    overall_error_scores = pd.read_csv(path + "/overall_error_scores.csv", index_col=0)
    std_error = pd.read_csv(path + "/std_error.csv", index_col=0)
    return pred, ind_error, overall_error_scores, std_error


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


#### Forecasting plotting


def print_single_forecast(train, test, prediction, y_label, model_label, color):
    fig, ax = plt.subplots(figsize=(15, 5), dpi=100)
    plt.locator_params(axis='x', nbins=5)
    ax.plot(train[len(train) - 168:], label='training')
    ax.plot(test[:168], label='actual', color="lightcoral")

    ax.plot(prediction[:168], label='forecast', color=color)
    ax.legend()
    ax.set(xlabel="Time", ylabel=y_label + " [Wh]", title=model_label + ": Forecast vs Actual")
    ax.grid(True, which='both')
    plt.show()


def print_double_forecast(train, test, pred1, pred2):
    fig, ax = plt.subplots(figsize=(15, 5), dpi=100)
    plt.locator_params(axis='x', nbins=5)
    ax.plot(train[len(train) - 24:], label='training')
    ax.plot(test[:48], label='actual', color="lightcoral")

    ax.plot(pred1[:48], label='LSTM forecast 24h', color="gold")
    ax.plot(pred2[:48], label='SVR forecast 24h', color="darkgreen")

    ax.legend()
    ax.set(xlabel="Time", ylabel="PV production [Wh]", title="Forecast vs Actual")
    ax.grid(True, which='both')
    plt.show()


def print_multi_forecast(train, test, predictions, colors, model_labels, y_label):
    fig, ax = plt.subplots(figsize=(15, 5), dpi=100)
    plt.locator_params(axis='x', nbins=5)
    # ax.plot(train[len(train) - 24:], label='training', color="rosybrown")
    ax.plot(test[48:72], label='training', color="rosybrown")  # 48:72
    ax.plot(test[72:120], label='actual', color="tomato", linestyle="--")  # 72:240
    for pred, mlabel, c in zip(predictions, model_labels, colors):
        ax.plot(pred[72:120], label=mlabel, color=c, alpha=0.7)  # 72:240
    ax.legend()
    ax.set(xlabel="Time", ylabel=y_label + " [Wh]", title="Forecast vs Actual")
    ax.grid(True, which='both')
    ax.set_ylim(top=20000)
    ax.set_ylim(bottom=-500)
    plt.show()


##### Error plotting

def plot_individual_score_exp3(lstm_predictions, lstm_individual_score, lstm_std_error):
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 8)

    for i, c in zip(range(0, 4), ["red", "blue", "green"]):
        error = lstm_std_error[i].loc["rmse"].item()
        ax.errorbar(x=lstm_individual_score[i]["rmse"].index, y=lstm_individual_score[i]["rmse"], yerr=error,
                    capsize=10,
                    errorevery=(3 * i, 22), color=c, fmt='-')

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
        ax.errorbar(x=x_labels, y=scores[i]["rmse"], yerr=error, color=c, fmt='o', capsize=10)

    ax.legend(["error1", "error2", "error3"])
    ax.set_ylabel("R-squared")
    ax.set_xlabel("Time")
    ax.grid(True, which='both')
    plt.show()


def print_error(error):
    fig, ax = plt.subplots(figsize=(15, 5), dpi=100)
    plt.locator_params(axis='x', nbins=5)
    ax.plot(error, label='error')

    ax.set(xlabel="Time", ylabel="RMSE [Wh]", title="Error plot")
    ax.grid(True, which='both')
    plt.show()


def plot_best_scores(scores, std_error, x_labels, ft_type, color):
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.text(s=ft_type + " forecasting: Winter", x=0.5, y=0.92, fontsize=14,
             ha='center', va='center')

    ax.errorbar(x=x_labels, y=scores, yerr=std_error, color=color, fmt='--o', label='error', capsize=10)
    ax.set_ylabel("Mean RMSE", labelpad=4)
    ax.set_xlabel("Model type")
    ax.legend()
    ax.grid(True, which='both')
    # ax.set_ylim(top=20000)
    #  ax.set_ylim(bottom=-500)
    #  plt.tight_layout()
    plt.show()


def plot_best_scores_together(scores, std_error, x_labels, ft_type, colors, labels):
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.text(s=ft_type + " forecasting", x=0.5, y=0.92, fontsize=14,
             ha='center', va='center')

    for score, std, color, label in zip(scores, std_error, colors, labels):
        ax.errorbar(x=x_labels, y=score, yerr=std, color=color, fmt='--o', label=label, capsize=10, alpha=0.7)
    ax.set_ylabel("Mean RMSE", labelpad=4)
    ax.set_xlabel("Month")
    ax.legend()
    ax.grid(True, which='both')
    #   ax.set_ylim(top=0.55)
    #   ax.set_ylim(bottom=-0.1)
    #  plt.tight_layout()
    plt.show()


def plot_best_r2scores_together(scores, x_labels, ft_type, colors, labels):
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.text(s=ft_type + " forecasting", x=0.5, y=0.92, fontsize=14,
             ha='center', va='center')

    for score, color, label in zip(scores, colors, labels):
        ax.errorbar(x=x_labels, y=score, color=color, fmt='--o', label=label, capsize=10, alpha=0.7)
    ax.set_ylabel("R2", labelpad=4)
    ax.set_xlabel("Month")
    ax.legend()
    ax.grid(True, which='both')
    #   ax.set_ylim(top=0.55)
    #   ax.set_ylim(bottom=-0.1)
    #  plt.tight_layout()
    plt.show()


#### Grid search plotting

# source: https://stackoverflow.com/questions/37161563/how-to-graph-grid-scores-from-gridsearchcv
# For plotting the results when tuning several hyperparameters, what I did was fixed all parameters to their best value
# except for one and plotted the mean score for the other parameter for each of its values.
#
def plot_grid_search_hyperparameter_opt(results, param_grid, best_params, title, model, ft_type):
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
    fig, ax = plt.subplots(1, len(param_grid), sharex='none', sharey='all', figsize=(15, 6))
    fig.text(s=ft_type + " forecast: Parameters of the " + model + " GridSearches during CV", x=0.5, y=0.95,
             fontsize=14, ha='center', va='center')
    fig.text(s=title, x=0.5, y=0.9, fontsize=12, ha='center', va='center')
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
        #    ax[i].set_ylim(top=1.1)
        #    ax[i].set_ylim(bottom=-0.9)
        ax[i].grid(True, which='both')
        ax[i].autoscale()
        loc = plticker.MultipleLocator(base=0.2)
        ax[i].yaxis.set_major_locator(loc)

    plt.legend()
    plt.show()


def plot_grid_search_best_scores(score1, score2, score3, horizon, model, ft_type):
    x_labels = []
    for h in horizon:
        x_labels.append(str(h) + "h")
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.text(s=ft_type + " forecast: Best scores of the " + model + " GridSearches", x=0.5, y=0.92, fontsize=14,
             ha='center', va='center')
    ax.errorbar(x=x_labels, y=score1, color="yellowgreen", fmt='--o', label="3 months")
    ax.errorbar(x=x_labels, y=score2, color="orange", fmt='--o', label="6 months")
    ax.errorbar(x=x_labels, y=score3, color="coral", fmt='--o', label="12 months")
    ax.set_ylabel("Mean R2 score", labelpad=4)
    ax.set_xlabel("Forecast horizon")
    ax.legend()
    ax.grid(True, which='both')
    ax.set_ylim(top=0.55)
    ax.set_ylim(bottom=-0.1)
    #  plt.tight_layout()
    plt.show()


def plot_grid_results(parent_folders, model, ft_type):
    horizon = [6, 12, 24]
    train_length = ["3mo", "6mo", "12mo"]
    results = []
    for i, ts in zip(range(len(horizon)), train_length):
        best_params, cv_results, param_grid, best_score = load_grid_search_results(parent_folders[i], model, horizon)
        results.append([ts, best_params, cv_results, param_grid, best_score])

        if model == "SVR":
            plot_grid_search_hyperparameter_opt(cv_results[0], param_grid[0], best_params[0],
                                                "Training set size: " + ts + ", Forecast horizon: 6h, 12h, and 24h",
                                                model, ft_type)
        else:
            for j, h in enumerate(["6h", "12h", "24h"]):
                plot_grid_search_hyperparameter_opt(cv_results[j], param_grid[j], best_params[j],
                                                    "Training set size: " + ts + ", Forecast horizon: " + h, model,
                                                    ft_type)
    plot_grid_search_best_scores(results[0][-1], results[1][-1], results[2][-1], horizon, model, ft_type)

    return results
