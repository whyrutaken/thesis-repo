from printer import *
from preparator import Preparator
from persistence_model import PersistenceModel
import tomli
import pandas as pd

with open("config.toml", mode="rb") as fp:
    config = tomli.load(fp)

svr = "SVR"
lstm = "LSTM"
arima = "ARIMA"
pv_path = "final-models/PV-forecast/"
cons_path = "final-models/Cons-forecast/"
pv = "solar_absolute"
cons = "demand_absolute"

# 6mo winter
train_from_date_winter = "2020-07-01 00:00"
test_from_date_winter = "2021-01-01 00:00"
test_to_date_winter = "2021-01-31 00:00"

# 6mo spring
train_from_date_spring = "2020-10-01 00:00"
test_from_date_spring = "2021-04-01 00:00"
test_to_date_spring = "2021-04-30 00:00"

# 6mo summer
train_from_date_summer = "2021-01-01 00:00"
test_from_date_summer = "2021-07-01 00:00"
test_to_date_summer = "2021-07-31 00:00"

# 6mo autumn
train_from_date_autumn = "2021-04-01 00:00"
test_from_date_autumn = "2021-10-01 00:00"
test_to_date_autumn = "2021-10-31 00:00"

# load PV models
pers_pv_winter = PersistenceModel(pv, train_from_date_winter, test_from_date_winter, test_to_date_winter)
pers_pv_spring = PersistenceModel(pv, train_from_date_spring, test_from_date_spring, test_to_date_spring)
pers_pv_summer = PersistenceModel(pv, train_from_date_summer, test_from_date_summer, test_to_date_summer)
pers_pv_autumn = PersistenceModel(pv, train_from_date_autumn, test_from_date_autumn, test_to_date_autumn)

arima_pv_winter = load_model_results_without_iteration(pv_path + arima + "/12-22--04-58-pv-arima-winter", arima,
                                                       horizon=6)
arima_pv_spring = load_model_results_without_iteration(pv_path + arima + "/12-24--21-54-pv-arima-spring", arima,
                                                       horizon=6)
arima_pv_summer = load_model_results_without_iteration(pv_path + arima + "/12-24--21-57-pv-arima-summer", arima,
                                                       horizon=6)
arima_pv_autumn = load_model_results_without_iteration(pv_path + arima + "/12-24--21-58-pv-arima-autumn", arima,
                                                       horizon=6)

svr_pv_winter = load_model_results_with_iteration(pv_path + svr + "/12-22--05-08-pv-svr-6mo-winter", svr, horizon=[6])
svr_pv_spring = load_model_results_with_iteration(pv_path + svr + "/12-22--05-11-pv-svr-6mo-spring", svr, horizon=[6])
svr_pv_summer = load_model_results_with_iteration(pv_path + svr + "/12-22--05-13-pv-svr-6mo-summer", svr, horizon=[6])
svr_pv_autumn = load_model_results_with_iteration(pv_path + svr + "/12-22--05-16-pv-svr-6mo-autumn", svr, horizon=[6])

lstm1_pv_winter = load_model_results_with_iteration(pv_path + lstm + "/12-26--04-31-pv-lstm1-winter", "LSTM1",
                                                    horizon=[6])
lstm1_pv_spring = load_model_results_with_iteration(pv_path + lstm + "/12-26--04-35-pv-lstm1-spring", "LSTM1",
                                                    horizon=[6])
lstm1_pv_summer = load_model_results_with_iteration(pv_path + lstm + "/12-26--04-39-pv-lstm1-summer", "LSTM1",
                                                    horizon=[6])
lstm1_pv_autumn = load_model_results_with_iteration(pv_path + lstm + "/12-26--04-41-pv-lstm1-autumn", "LSTM1",
                                                    horizon=[6])

lstm2_pv_winter = load_model_results_with_iteration(pv_path + lstm + "/12-26--04-53-pv-lstm2-winter", "LSTM2",
                                                    horizon=[6])
lstm2_pv_spring = load_model_results_with_iteration(pv_path + lstm + "/12-26--04-51-pv-lstm2-spring", "LSTM2",
                                                    horizon=[6])
lstm2_pv_summer = load_model_results_with_iteration(pv_path + lstm + "/12-26--04-48-pv-lstm2-summer", "LSTM2",
                                                    horizon=[6])
lstm2_pv_autumn = load_model_results_with_iteration(pv_path + lstm + "/12-26--04-46-pv-lstm2-autumn", "LSTM2",
                                                    horizon=[6])

# load consumption models
pers_cons_winter = PersistenceModel(cons, train_from_date_winter, test_from_date_winter, test_to_date_winter)
pers_cons_spring = PersistenceModel(cons, train_from_date_spring, test_from_date_spring, test_to_date_spring)
pers_cons_summer = PersistenceModel(cons, train_from_date_summer, test_from_date_summer, test_to_date_summer)
pers_cons_autumn = PersistenceModel(cons, train_from_date_autumn, test_from_date_autumn, test_to_date_autumn)

arima_cons_winter = load_model_results_without_iteration(cons_path + arima + "/12-24--22-04-cp-arima-winter", arima,
                                                         horizon=6)
arima_cons_spring = load_model_results_without_iteration(cons_path + arima + "/12-24--22-03-cp-arima-spring", arima,
                                                         horizon=6)
arima_cons_summer = load_model_results_without_iteration(cons_path + arima + "/12-24--22-02-cp-arima-summer", arima,
                                                         horizon=6)
arima_cons_autumn = load_model_results_without_iteration(cons_path + arima + "/12-24--22-00-cp-arima-autumn", arima,
                                                         horizon=6)

svr_cons_winter = load_model_results_with_iteration(cons_path + svr + "/12-23--08-08-cp-svr-winter", svr, horizon=[6])
svr_cons_spring = load_model_results_with_iteration(cons_path + svr + "/12-23--08-07-cp-svr-spring", svr, horizon=[6])
svr_cons_summer = load_model_results_with_iteration(cons_path + svr + "/12-23--08-06-cp-svr-summer", svr, horizon=[6])
svr_cons_autumn = load_model_results_with_iteration(cons_path + svr + "/12-23--08-04-cp-svr-autumn", svr, horizon=[6])

lstm1_cons_winter = load_model_results_with_iteration(cons_path + lstm + "/12-27--05-01-cp-lstm1-winter", "LSTM1",
                                                      horizon=[6])
lstm1_cons_spring = load_model_results_with_iteration(cons_path + lstm + "/12-27--05-00-cp-lstm1-spring", "LSTM1",
                                                      horizon=[6])
lstm1_cons_summer = load_model_results_with_iteration(cons_path + lstm + "/12-27--04-58-cp-lstm1-summer", "LSTM1",
                                                      horizon=[6])
lstm1_cons_autumn = load_model_results_with_iteration(cons_path + lstm + "/12-27--04-59-cp-lstm1-autumn", "LSTM1",
                                                      horizon=[6])

lstm2_cons_winter = load_model_results_with_iteration(cons_path + lstm + "/12-27--04-53-cp-lstm2-winter", "LSTM2",
                                                      horizon=[6])
lstm2_cons_spring = load_model_results_with_iteration(cons_path + lstm + "/12-27--04-54-cp-lstm2-spring", "LSTM2",
                                                      horizon=[6])
lstm2_cons_summer = load_model_results_with_iteration(cons_path + lstm + "/12-27--04-55-cp-lstm2-summer", "LSTM2",
                                                      horizon=[6])
lstm2_cons_autumn = load_model_results_with_iteration(cons_path + lstm + "/12-27--04-56-cp-lstm2-autumn", "LSTM2",
                                                      horizon=[6])

# %%
colors = ["black", "sandybrown", "limegreen", "dodgerblue", "hotpink"]
model_labels = ["Persistence", "ARIMA", "SVR", "LSTM1", "LSTM2"]

# plot PV predictions for different months
predictions_winter = [pers_pv_winter.prediction, arima_pv_winter[0], svr_pv_winter[0][0][0], lstm1_pv_winter[0][0][0],
                      lstm2_pv_winter[0][0][0]]
print_multi_forecast(pers_pv_winter.train, pers_pv_winter.test, predictions_winter, colors, model_labels, 'PV power')

predictions_spring = [pers_pv_spring.prediction, arima_pv_spring[0], svr_pv_spring[0][0][0], lstm1_pv_spring[0][0][0],
                      lstm2_pv_spring[0][0][0]]
print_multi_forecast(pers_pv_spring.train, pers_pv_spring.test, predictions_spring, colors, model_labels, 'PV power')

predictions_summer = [pers_pv_summer.prediction, arima_pv_summer[0], svr_pv_summer[0][0][0], lstm1_pv_summer[0][0][0],
                      lstm2_pv_summer[0][0][0]]
print_multi_forecast(pers_pv_summer.train, pers_pv_summer.test, predictions_summer, colors, model_labels, 'PV power')

predictions_autumn = [pers_pv_autumn.prediction, arima_pv_autumn[0], svr_pv_autumn[0][0][0], lstm1_pv_autumn[0][0][0],
                      lstm2_pv_autumn[0][0][0]]
print_multi_forecast(pers_pv_autumn.train, pers_pv_autumn.test, predictions_autumn, colors, model_labels, 'PV power')

# %%

# plot cons predictions for different months
predictions_cons_winter = [pers_cons_winter.prediction, arima_cons_winter[0], svr_cons_winter[0][0][0],
                           lstm1_cons_winter[0][0][0],
                           lstm2_cons_winter[0][0][0]]
print_multi_forecast(pers_cons_winter.train, pers_cons_winter.test, predictions_cons_winter, colors, model_labels,
                     'Power consumption')

predictions_cons_spring = [pers_cons_spring.prediction, arima_cons_spring[0], svr_cons_spring[0][0][0],
                           lstm1_cons_spring[0][0][0],
                           lstm2_cons_spring[0][0][0]]
print_multi_forecast(pers_cons_spring.train, pers_cons_spring.test, predictions_cons_spring, colors, model_labels,
                     'Power consumption')

predictions_cons_summer = [pers_cons_summer.prediction, arima_cons_summer[0], svr_cons_summer[0][0][0],
                           lstm1_cons_summer[0][0][0],
                           lstm2_cons_summer[0][0][0]]
print_multi_forecast(pers_cons_summer.train, pers_cons_summer.test, predictions_cons_summer, colors, model_labels,
                     'Power consumption')

predictions_cons_autumn = [pers_cons_autumn.prediction, arima_cons_autumn[0], svr_cons_autumn[0][0][0],
                           lstm1_cons_autumn[0][0][0],
                           lstm2_cons_autumn[0][0][0]]
print_multi_forecast(pers_cons_autumn.train, pers_cons_autumn.test, predictions_cons_autumn, colors, model_labels,
                     'Power consumption')

# %%

# gather PV error scores
pers_scores = [pers_pv_winter.overall_scores["rmse"][0], pers_pv_spring.overall_scores["rmse"][0],
               pers_pv_summer.overall_scores["rmse"][0], pers_pv_autumn.overall_scores["rmse"][0]]
pers_r2scores = [pers_pv_winter.overall_scores["r2"][0], pers_pv_spring.overall_scores["r2"][0],
                 pers_pv_summer.overall_scores["r2"][0], pers_pv_autumn.overall_scores["r2"][0]]
pers_std = [pers_pv_winter.std_error[0], pers_pv_spring.std_error[0], pers_pv_summer.std_error[0],
            pers_pv_autumn.std_error[0]]

arima_scores = [arima_pv_winter[2]["rmse"][0], arima_pv_spring[2]["rmse"][0], arima_pv_summer[2]["rmse"][0],
                arima_pv_autumn[2]["rmse"][0]]
arima_r2scores = [arima_pv_winter[2]["r2"][0], arima_pv_spring[2]["r2"][0], arima_pv_summer[2]["r2"][0],
                  arima_pv_autumn[2]["r2"][0]]
arima_std = [arima_pv_winter[3].iloc[0, 0], arima_pv_spring[3].iloc[0, 0], arima_pv_summer[3].iloc[0, 0],
             arima_pv_autumn[3].iloc[0, 0]]

svr_scores = [svr_pv_winter[0][2][0]["rmse"][0], svr_pv_spring[0][2][0]["rmse"][0], svr_pv_summer[0][2][0]["rmse"][0],
              svr_pv_autumn[0][2][0]["rmse"][0]]
svr_r2scores = [svr_pv_winter[0][2][0]["r2"][0], svr_pv_spring[0][2][0]["r2"][0], svr_pv_summer[0][2][0]["r2"][0],
                svr_pv_autumn[0][2][0]["r2"][0]]
svr_std = [svr_pv_winter[0][3][0].iloc[0, 0], svr_pv_spring[0][3][0].iloc[0, 0], svr_pv_summer[0][3][0].iloc[0, 0],
           svr_pv_autumn[0][3][0].iloc[0, 0]]

lstm1_scores = [lstm1_pv_winter[0][2][0]["rmse"][0], lstm1_pv_spring[0][2][0]["rmse"][0],
                lstm1_pv_summer[0][2][0]["rmse"][0], lstm1_pv_autumn[0][2][0]["rmse"][0]]
lstm1_r2scores = [lstm1_pv_winter[0][2][0]["r2"][0], lstm1_pv_spring[0][2][0]["r2"][0],
                  lstm1_pv_summer[0][2][0]["r2"][0], lstm1_pv_autumn[0][2][0]["r2"][0]]
lstm1_std = [lstm1_pv_winter[0][3][0].iloc[0, 0], lstm1_pv_spring[0][3][0].iloc[0, 0],
             lstm1_pv_summer[0][3][0].iloc[0, 0], lstm1_pv_autumn[0][3][0].iloc[0, 0]]

lstm2_scores = [lstm2_pv_winter[0][2][0]["rmse"][0], lstm2_pv_spring[0][2][0]["rmse"][0],
                lstm2_pv_summer[0][2][0]["rmse"][0], lstm2_pv_autumn[0][2][0]["rmse"][0]]
lstm2_r2scores = [lstm2_pv_winter[0][2][0]["r2"][0], lstm2_pv_spring[0][2][0]["r2"][0],
                  lstm2_pv_summer[0][2][0]["r2"][0], lstm2_pv_autumn[0][2][0]["r2"][0]]
lstm2_std = [lstm2_pv_winter[0][3][0].iloc[0, 0], lstm2_pv_spring[0][3][0].iloc[0, 0],
             lstm2_pv_summer[0][3][0].iloc[0, 0], lstm2_pv_autumn[0][3][0].iloc[0, 0]]

# print PV scores
labels = ["January", "April", "July", "October"]
scores_rmse = [pers_scores, arima_scores, svr_scores, lstm1_scores, lstm2_scores]
scores_r2 = [pers_r2scores, arima_r2scores, svr_r2scores, lstm1_r2scores, lstm2_r2scores]
std_error = [pers_std, arima_std, svr_std, lstm1_std, lstm2_std]
plot_best_scores_together(scores_rmse, std_error, labels, "PV power", colors, model_labels)
plot_best_r2scores_together(scores_r2, labels, "PV power", colors, model_labels)

#%%
# gather consumption error scores
pers_cons_scores = [pers_cons_winter.overall_scores["rmse"][0], pers_cons_spring.overall_scores["rmse"][0],
                    pers_cons_summer.overall_scores["rmse"][0], pers_cons_autumn.overall_scores["rmse"][0]]
pers_cons_r2scores = [pers_cons_winter.overall_scores["r2"][0], pers_cons_spring.overall_scores["r2"][0],
                      pers_cons_summer.overall_scores["r2"][0], pers_cons_autumn.overall_scores["r2"][0]]
pers_cons_std = [pers_cons_winter.std_error[0], pers_cons_spring.std_error[0], pers_cons_summer.std_error[0],
                 pers_cons_autumn.std_error[0]]

arima_cons_scores = [arima_cons_winter[2]["rmse"][0], arima_cons_spring[2]["rmse"][0], arima_cons_summer[2]["rmse"][0],
                     arima_cons_autumn[2]["rmse"][0]]
arima_cons_r2scores = [arima_cons_winter[2]["r2"][0], arima_cons_spring[2]["r2"][0], arima_cons_summer[2]["r2"][0],
                       arima_cons_autumn[2]["r2"][0]]
arima_cons_std = [arima_cons_winter[3].iloc[0, 0], arima_cons_spring[3].iloc[0, 0], arima_cons_summer[3].iloc[0, 0],
                  arima_cons_autumn[3].iloc[0, 0]]

svr_cons_scores = [svr_cons_winter[0][2][0]["rmse"][0], svr_cons_spring[0][2][0]["rmse"][0],
                   svr_cons_summer[0][2][0]["rmse"][0],
                   svr_cons_autumn[0][2][0]["rmse"][0]]
svr_cons_r2scores = [svr_cons_winter[0][2][0]["r2"][0], svr_cons_spring[0][2][0]["r2"][0],
                     svr_cons_summer[0][2][0]["r2"][0],
                     svr_cons_autumn[0][2][0]["r2"][0]]
svr_cons_std = [svr_cons_winter[0][3][0].iloc[0, 0], svr_cons_spring[0][3][0].iloc[0, 0],
                svr_cons_summer[0][3][0].iloc[0, 0],
                svr_cons_autumn[0][3][0].iloc[0, 0]]

lstm1_cons_scores = [lstm1_cons_winter[0][2][0]["rmse"][0], lstm1_cons_spring[0][2][0]["rmse"][0],
                     lstm1_cons_summer[0][2][0]["rmse"][0], lstm1_cons_autumn[0][2][0]["rmse"][0]]
lstm1_cons_r2scores = [lstm1_cons_winter[0][2][0]["r2"][0], lstm1_cons_spring[0][2][0]["r2"][0],
                       lstm1_cons_summer[0][2][0]["r2"][0], lstm1_cons_autumn[0][2][0]["r2"][0]]
lstm1_cons_std = [lstm1_cons_winter[0][3][0].iloc[0, 0], lstm1_cons_spring[0][3][0].iloc[0, 0],
                  lstm1_cons_summer[0][3][0].iloc[0, 0], lstm1_cons_autumn[0][3][0].iloc[0, 0]]

lstm2_cons_scores = [lstm2_cons_winter[0][2][0]["rmse"][0], lstm2_cons_spring[0][2][0]["rmse"][0],
                     lstm2_cons_summer[0][2][0]["rmse"][0], lstm2_cons_autumn[0][2][0]["rmse"][0]]
lstm2_cons_r2scores = [lstm2_cons_winter[0][2][0]["r2"][0], lstm2_cons_spring[0][2][0]["r2"][0],
                       lstm2_cons_summer[0][2][0]["r2"][0], lstm2_cons_autumn[0][2][0]["r2"][0]]
lstm2_cons_std = [lstm2_cons_winter[0][3][0].iloc[0, 0], lstm2_cons_spring[0][3][0].iloc[0, 0],
                  lstm2_cons_summer[0][3][0].iloc[0, 0], lstm2_cons_autumn[0][3][0].iloc[0, 0]]

# print consumption scores
labels = ["January", "April", "July", "October"]
scores_cons_rmse = [pers_cons_scores, arima_cons_scores, svr_cons_scores, lstm1_cons_scores, lstm2_cons_scores]
scores_cons_r2 = [pers_cons_r2scores, arima_cons_r2scores, svr_cons_r2scores, lstm1_cons_r2scores, lstm2_cons_r2scores]
std_cons_error = [pers_cons_std, arima_cons_std, svr_cons_std, lstm1_cons_std, lstm2_cons_std]
plot_best_scores_together(scores_cons_rmse, std_cons_error, labels, "Power consumption", colors, model_labels)
plot_best_r2scores_together(scores_cons_r2, labels, "Power consumption", colors, model_labels)
# %%
test_winter = pers_pv_winter.test.iloc[:len(pers_pv_winter.prediction)]["solar_absolute"]
test_spring = pers_pv_spring.test.iloc[:len(pers_pv_spring.prediction)]["solar_absolute"]
test_summer = pers_pv_summer.test.iloc[:len(pers_pv_summer.prediction)]["solar_absolute"]
test_autumn = pers_pv_autumn.test.iloc[:len(pers_pv_autumn.prediction)]["solar_absolute"]

# %%


pers_residuals_winter = test_winter - pers_pv_winter.prediction
pers_residuals_spring = test_spring - pers_pv_spring.prediction
pers_residuals_summer = test_summer - pers_pv_summer.prediction
pers_residuals_autumn = test_autumn - pers_pv_autumn.prediction

ax = pers_residuals_winter.plot(kind="hist")
pers_residuals_winter.plot(kind="kde", ax=ax, secondary_y=True)
plt.ylabel("Residuals")
plt.show()

ax = pers_residuals_spring.plot(kind="hist")
pers_residuals_spring.plot(kind="kde", ax=ax, secondary_y=True)
plt.ylabel("Residuals")
plt.show()

aee = arima_pv_winter[0]
arima_residuals_winter = test_winter - arima_pv_winter[0]["0"]
arima_residuals_spring = test_spring - arima_pv_spring[0]["0"]
arima_residuals_summer = test_summer - arima_pv_summer[0]["0"]
arima_residuals_autumn = test_autumn - arima_pv_autumn[0]["0"]

ax = arima_residuals_winter.plot(kind="hist")
arima_residuals_winter.plot(kind="kde", ax=ax, secondary_y=True)
plt.ylabel("Residuals")
plt.show()

svr_residuals_winter = test_winter - svr_pv_winter[0][0][0]["0"]
svr_residuals_spring = test_spring - svr_pv_spring[0][0][0]["0"]
svr_residuals_summer = test_summer - svr_pv_summer[0][0][0]["0"]
svr_residuals_autumn = test_autumn - svr_pv_autumn[0][0][0]["0"]

ax = svr_residuals_winter.plot(kind="hist")
svr_residuals_winter.plot(kind="kde", ax=ax, secondary_y=True)
plt.ylabel("Residuals")
plt.show()

lstm1_residuals_winter = test_winter - lstm1_pv_winter[0][0][0]["0"]
lstm1_residuals_spring = test_spring - lstm1_pv_spring[0][0][0]["0"]
lstm1_residuals_summer = test_summer - lstm1_pv_summer[0][0][0]["0"]
lstm1_residuals_autumn = test_autumn - lstm1_pv_autumn[0][0][0]["0"]

ax = lstm1_residuals_winter.plot(kind="hist")
lstm1_residuals_winter.plot(kind="kde", ax=ax, secondary_y=True)
plt.ylabel("Residuals")
plt.show()

ax = lstm1_residuals_summer.plot(kind="hist")
lstm1_residuals_summer.plot(kind="kde", ax=ax, secondary_y=True)
plt.ylabel("Residuals")
plt.show()

lstm2_residuals_winter = test_winter - lstm2_pv_winter[0][0][0]["0"]
lstm2_residuals_spring = test_spring - lstm2_pv_spring[0][0][0]["0"]
lstm2_residuals_summer = test_summer - lstm2_pv_summer[0][0][0]["0"]
lstm2_residuals_autumn = test_autumn - lstm2_pv_autumn[0][0][0]["0"]
