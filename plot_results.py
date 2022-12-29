#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import deepdish as dd
import matplotlib.ticker as plticker
import printer
from preparator import Preparator
from persistence_model import PersistenceModel
import tomli

with open("config.toml", mode="rb") as fp:
    config = tomli.load(fp)

persistence = PersistenceModel(config)
printer.print_single_forecast(persistence.train, persistence.test, persistence.prediction, y_label="PV power", model_label="Persistence model")
#%%





def load_training_data(parent_folder, model_name, horizon):
    predictions = []
    individual_error_scores = []
    overall_error_scores = []
    std_error = []

    horizon_=[]
    for h in horizon:
        iteration = []
        for i in range(1, 5):
            path = parent_folder + "/models-i" + str(i) + "/" + model_name + "-" + str(h) + "h"
            pred = pd.read_csv(path + "/predictions.csv", parse_dates=True, index_col=0, names=["predictions"])
            pred.index = pd.DatetimeIndex(pred.index)
            predictions.append(pred)
            ind_error = pd.read_csv(path + "/individual_error_scores.csv", parse_dates=True, index_col=0)
            ind_error.index = pd.DatetimeIndex(ind_error.index)
            individual_error_scores.append(ind_error)
         #   overall_error_scores.append(pd.read_csv(path + "/overall_error_scores.csv", index_col=0))
            overall_error=pd.read_csv(path + "/overall_error_scores.csv", index_col=0)
            std_error=pd.read_csv(path + "/std_error.csv", index_col=0, names=["std_error"])
      #      std_error.append( pd.read_csv(path + "/std_error.csv", index_col=0))
            iteration.append([pred, ind_error, overall_error, std_error])
        horizon_.append(iteration)


 #   array = np.array([predictions, individual_error_scores, overall_error_scores, std_error]).T
    #return [predictions, individual_error_scores, overall_error_scores, std_error]
    # horizon -> iteration -> [pred, ind_error, overall_error, std_error]
    return horizon_

horizon = [6, 12, 24]
svr = "SVR"
lstm = "LSTM"
arima = "ARIMA"
pv_path = "final-models/PV-forecast/"
cons_path = "final-models/Cons-forecast/"
svr_pv_path = pv_path + "SVR/"
svr_cons_path = cons_path + "SVR/"
lstm_pv_path = pv_path + "LSTM/"
lstm_cons_path = cons_path + "LSTM/"
arima_pv_path = pv_path + "ARIMA/"
arima_cons_path = cons_path + "ARIMA/"

train_from_date = "2020-10-01 00:00"
test_from_date = "2021-01-01 00:00"
test_to_date = "2021-10-31 00:00"
attribute = "solar_absolute"
preparator = Preparator(attribute, train_from_date=train_from_date, test_from_date=test_from_date)
y_train, y_test = preparator.y_train, preparator.y_test


#%%
horizon = [6]
train_from_date = "2020-07-01 00:00"
test_from_date = "2021-01-01 00:00"
test_to_date = "2021-01-31 00:00"

attribute = "solar_absolute"
preparator = Preparator(attribute, train_from_date=train_from_date, test_from_date=test_from_date)
y_train, y_test = preparator.y_train, preparator.y_test
svr_pv_6mo = load_training_data(svr_pv_path+"12-22--05-08-pv-svr-6mo-winter", svr, horizon)

print(svr_pv_6mo[0][0][0].equals(svr_pv_6mo[0][1][0]))
#svr_pv_6mo = load_training_data(svr_pv_path+"12-22--04-58-pv-arima-6mo-winter", svr, horizon)
#svr_pv_12mo = load_training_data(svr_pv_path+"12-22--04-59-pv-arima-12mo-winter", svr, horizon)

printer.print_error(svr_pv_6mo[0][0][0])
printer.print_single_forecast(y_train, y_test, svr_pv_6mo[0][0][0])

#%%
train_from_date = "2020-10-01 00:00"
test_from_date = "2021-01-01 00:00"
test_to_date = "2021-10-31 00:00"
attribute = "solar_absolute"
preparator = Preparator(attribute, train_from_date=train_from_date, test_from_date=test_from_date)
y_train, y_test = preparator.y_train, preparator.y_test
arima_pv_3mo = load_training_data(arima_pv_path+"12-22--04-52-pv-arima-3mo-winter", arima, horizon)
arima_pv_6mo = load_training_data(arima_pv_path+"12-22--04-58-pv-arima-6mo-winter", arima, horizon)
arima_pv_12mo = load_training_data(arima_pv_path+"12-22--04-59-pv-arima-12mo-winter", arima, horizon)



#%%
print(arima_pv_12mo[0][0][0].equals(arima_pv_12mo[0][1][0]))
print(arima_pv_6mo[0][0][0].equals(arima_pv_6mo[0][2][0]))
print(arima_pv_6mo[0][0][0].equals(arima_pv_6mo[0][3][0]))

#%%
printer.print_error(arima_pv_3mo[0][0])
printer.print_single_forecast(y_train, y_test, arima_pv_3mo[0][0])