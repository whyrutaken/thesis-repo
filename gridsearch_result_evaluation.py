#####
#   Script for plotting the grid search results of the models
#

from printer import *



#%%
svr_pv_grid_path = "final-models/PV-forecast/SVR/gridsearch/"
svr_cons_grid_path = "final-models/Cons-forecast/SVR/gridsearch/"
lstm_pv_grid_path = "final-models/PV-forecast/LSTM/gridsearch/"
lstm_cons_grid_path = "final-models/Cons-forecast/LSTM/gridsearch/"
svr = "SVR"
lstm = "LSTM"
pv = "PV"
cons = "Consumption"


lstm2_test = plot_grid_results(["12-23--02-34"], lstm, cons)

svr1y_test = plot_grid_results(["12-23--03-08"], svr, pv)

#%%
lstm2_test2 = plot_grid_results(["12-22--20-27-lstm-test-val-set"], lstm, cons)


#%%
parent_folders = [svr_pv_grid_path + "12-24--06-20-pv-svr-3mo-grid", svr_pv_grid_path + "12-17--22-39-pv-svr-6mo-grid", svr_pv_grid_path + "12-18--00-33-pv-svr-1y-grid"]
svr_pv = plot_grid_results(parent_folders, svr, pv)
#%%
parent_folders = [svr_cons_grid_path + "12-21--00-50-cf-svr-3m-grid", svr_cons_grid_path + "12-21--00-49-cf-svr-6m-grid", svr_cons_grid_path + "12-21--00-48-cf-svr-1y-grid"]
svr_cons = plot_grid_results(parent_folders, svr, cons)

#%%
parent_folders = [lstm_pv_grid_path+"12-20--23-41-pv-lstm1-3mo-grid", lstm_pv_grid_path+"12-19--20-41-pv-lstm1-6mo-grid", lstm_pv_grid_path+"12-20--03-38-pv-lstm1-12-mo-grid"]
lstm1_pv = plot_grid_results(parent_folders, lstm, pv)

#%%
parent_folders = [lstm_cons_grid_path+"12-20--22-57-cf-lstm1-3mo-grid", lstm_cons_grid_path+"12-20--04-22-cf-lstm1-6mo-grid", lstm_cons_grid_path+"12-20--16-49-cf-lstm1-1y-grid"]
lstm1_cons = plot_grid_results(parent_folders, lstm, cons)

#%%
parent_folders = [lstm_pv_grid_path+"12-21--01-18-pv-lstm2-3mo-grid", lstm_pv_grid_path+"12-21--01-15-pv-lstm2-6mo-grid", lstm_pv_grid_path+"12-21--01-12-pv-lstm2-12mo-grid"]
lstm2_pv = plot_grid_results(parent_folders, lstm, pv)

#%%
parent_folders = [lstm_cons_grid_path+"12-21--01-08-cf-lstm2-3mo-grid", lstm_cons_grid_path+"12-21--01-09-cf-lstm2-6mo-grid", lstm_cons_grid_path+"12-21--01-10-cf-lstm2-12mo-grid"]
lstm2_cons = plot_grid_results(parent_folders, lstm, cons)



# %%
parent_folder = "12-16--06-35-arima-winter"
# lstm_predictions, lstm_individual_error, lstm_overall_error, lstm_std_error = load_training_data(parent_folder, "LSTM", horizon)
arima_predictions, arima_individual_error, arima_overall_error, arima_std_error = load_model_results(parent_folder,
                                                                                                     "ARIMA", horizon)
# svr_predictions, svr_individual_error, svr_overall_error, svr_std_error = load_training_data(parent_folder, "SVR", horizon)
# %%
plot_individual_score_exp3(arima_predictions, arima_individual_error, arima_std_error)
plot_overall_score_exp1(arima_overall_error, arima_std_error, horizon)
