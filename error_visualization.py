import pandas as pd
import matplotlib.pyplot as plt


def read_data(parent_folder, model_name, horizon):
    predictions = []
    individual_error_scores = []
    overall_error_scores = []
    std_error = []
    for h in horizon:
        for i in range(1, 2):
            pred = pd.read_csv(
                parent_folder + "/models-i" + str(i) + "/" + model_name + "-" + str(h) + "h/predictions.csv",
                parse_dates=True, index_col=0)
            pred.index = pd.DatetimeIndex(pred.index)
            predictions.append(pred)
            ind_error = pd.read_csv(parent_folder + "/models-i" + str(i) + "/" + model_name + "-" + str(
                h) + "h/individual_error_scores.csv", parse_dates=True, index_col=0)
            ind_error.index = pd.DatetimeIndex(ind_error.index)
            individual_error_scores.append(ind_error)
            overall_error_scores.append(pd.read_csv(
                parent_folder + "/models-i" + str(i) + "/" + model_name + "-" + str(h) + "h/overall_error_scores.csv",
                index_col=0))
            std_error.append(
                pd.read_csv(parent_folder + "/models-i" + str(i) + "/" + model_name + "-" + str(h) + "h/std_error.csv",
                            index_col=0))
    return predictions, individual_error_scores, overall_error_scores, std_error


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
        ax.errorbar(x=lstm_individual_score[i]["rmse"].index, y=lstm_individual_score[i]["rmse"], yerr=error, capsize=10,
                    errorevery=(3 * i, 22), color=c, fmt='-')
    #    ax.errorbar(x=lstm_predictions[i].index, y=lstm_predictions[i].iloc[:,0], yerr=error, errorevery=(3 * i, 22), color=c, fmt='-')

    ax.legend(["error1", "error2", "error3"])
    ax.set_ylabel("RMSE [Wh]")
    ax.set_xlabel("Time")
    ax.grid(True, which='both')
    plt.show()


def plot_overall_score_exp1(lstm_overall_scores, lstm_std_error):
    fig, ax = plt.subplots()
    for i, c in zip(range(0, 1), ["red", "blue", "green"]):
        error = lstm_std_error[i].loc["rmse"].item()
        #  lstm_predictions[i].plot(use_index=True, ax=ax)
        #    lstm_individual_score[i]["rmse"].plot(use_index=True, ax=ax, color=c)
        ax.errorbar(x=["12h"], y=lstm_overall_scores[i]["rmse"], yerr=error, color=c, fmt='o', capsize=10)
    #    ax.errorbar(x=lstm_predictions[i].index, y=lstm_predictions[i].iloc[:,0], yerr=error, errorevery=(3 * i, 22), color=c, fmt='-')

    ax.legend(["error1", "error2", "error3"])
    ax.set_ylabel("RMSE [Wh]")
    ax.set_xlabel("Time")
    ax.grid(True, which='both')
    plt.show()


parent_folder = "12-09--00-19"
horizon = [12]
lstm_predictions, lstm_individual_error, lstm_overall_error, lstm_std_error = read_data(parent_folder, "LSTM", horizon)
arima_predictions, arima_individual_error, arima_overall_error, arima_std_error = read_data(parent_folder, "ARIMA",
                                                                                            horizon)
svr_predictions, svr_individual_error, svr_overall_error, svr_std_error = read_data(parent_folder, "SVR", horizon)

#plot_individual_score_exp3(lstm_predictions, lstm_individual_error, lstm_std_error)
plot_overall_score_exp1(lstm_overall_error, lstm_std_error)
