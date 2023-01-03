#####
#   Script for plotting and comparing the baseline models
#

from printer import *
from preparator import Preparator
from persistence_model import PersistenceModel
import tomli
import pandas as pd



with open("config.toml", mode="rb") as fp:
    config = tomli.load(fp)


#print_single_forecast(persistence.train, persistence.test, persistence.prediction, y_label="PV power", model_label="Persistence model", color="slategrey")

horizon = [6,12,24]
arima = "ARIMA"
arima_pv_path = "final-models/PV-forecast/ARIMA/"
arima_cons_path = "final-models/Cons-forecast/ARIMA/"
persistence = PersistenceModel(config)
#%%

# model iteration -> [predictions, individual_error_Scores, overall_error_scores, std_error] -> forecast horizon
arima_ = load_model_results_with_iteration(arima_pv_path + "12-22--04-58-pv-arima-6mo-winter", arima, horizon)


# arima PV train size comparison
arima_3 = load_model_results_with_iteration(arima_pv_path + "12-22--04-52-pv-arima-3mo-winter", arima, horizon)
arima_6 = load_model_results_with_iteration(arima_pv_path + "12-22--04-58-pv-arima-6mo-winter", arima, horizon)
arima_12 = load_model_results_with_iteration(arima_pv_path + "12-22--04-59-pv-arima-12mo-winter", arima, horizon)
colors = ["dodgerblue", "hotpink", "limegreen"]
model_labels = ["ARIMA 3mo", "ARIMA 6mo", "ARIMA 12mo"]

# horizon = 6h
predictions_train_size_6h = [arima_3[0][0][0], arima_6[0][0][0], arima_12[0][0][0]]
print_multi_forecast(persistence.train, persistence.test, predictions_train_size_6h, y_label="PV power", model_labels=model_labels, colors=colors)

# horizon = 12h
predictions_train_size_12h = [arima_3[0][0][1], arima_6[0][0][1], arima_12[0][0][1]]
print_multi_forecast(persistence.train, persistence.test, predictions_train_size_12h, y_label="PV power", model_labels=model_labels, colors=colors)

# horizon = 24h
predictions_train_size_24h = [arima_3[0][0][2], arima_6[0][0][2], arima_12[0][0][2]]
print_multi_forecast(persistence.train, persistence.test, predictions_train_size_24h, y_label="PV power", model_labels=model_labels, colors=colors)


#%%
# arima CP train size comparison
arima_cp_3 = load_model_results_with_iteration(arima_cons_path + "12-26--04-08-cp-arima-3mo-winter", arima, horizon)
arima_cp_6 = load_model_results_with_iteration(arima_cons_path + "12-26--04-07-cp-arima-6mo-winter", arima, horizon)
arima_cp_12 = load_model_results_with_iteration(arima_cons_path + "12-26--04-06-cp-arima-12mo-winter", arima, horizon)
colors = ["dodgerblue", "hotpink", "limegreen"]
model_labels = ["ARIMA 3mo", "ARIMA 6mo", "ARIMA 12mo"]


# horizon = 6h
predictions_train_size_6h = [arima_cp_3[0][0][0], arima_cp_6[0][0][0], arima_cp_12[0][0][0]]
print_multi_forecast(persistence.train, persistence.test, predictions_train_size_6h, y_label="Power consumption", model_labels=model_labels, colors=colors)
#%%
# horizon = 12h
predictions_train_size_12h = [arima_cp_3[0][0][1], arima_cp_6[0][0][1], arima_cp_12[0][0][1]]
print_multi_forecast(persistence.train, persistence.test, predictions_train_size_12h, y_label="Power consumption", model_labels=model_labels, colors=colors)
#%%
# horizon = 24h
predictions_train_size_24h = [arima_cp_3[0][0][2], arima_cp_6[0][0][2], arima_cp_12[0][0][2]]
print_multi_forecast(persistence.train, persistence.test, predictions_train_size_24h, y_label="Power consumption", model_labels=model_labels, colors=colors)


#%%

## baseline model comparison
## PV
colors = ["dodgerblue", "limegreen"]
model_labels = ["ARIMA 6h", "Persistence"]
predictions_baseline = [arima_6[0][0][0], persistence.prediction]
print_multi_forecast(persistence.train, persistence.test, predictions_baseline, y_label="PV power", model_labels=model_labels, colors=colors)

#%%
## CP
horizon=6
arima_cp6 = load_model_results_without_iteration(arima_cons_path + "12-24--22-04-cp-arima-winter", arima, horizon)
predictions_baseline_cp = [arima_cp6[0], persistence.prediction]
print_multi_forecast(persistence.train, persistence.test, predictions_baseline_cp, y_label="Power consumption", model_labels=model_labels, colors=colors)




print(arima_6[0][0][0].equals(arima_12[0][0][0]))

