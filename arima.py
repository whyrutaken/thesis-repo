#%%
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot as plt
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#["2020-06-01":"2020-09-03"]
#%%
class ARIMAModel:
    def __init__(self, attribute, from_date, to_date):
        self.master_df = pd.read_csv("extracted-data/master-df.csv")
        self.df = self.set_attribute_and_dates(attribute, from_date, to_date)
        self.df = self.df.reset_index(drop=True)

    def set_attribute_and_dates(self, attr, date1, date2):
        return self.master_df[date1:date2].loc[:, attr]
#%%
master_df = pd.read_csv("extracted-data/master-df.csv")
solar_df = master_df.solar_absolute
solar_df = solar_df.reset_index(drop=True)
load_df = master_df.demand_absolute["2021-01-01":"2022-01-01"]
autocorrelation_plot(master_df.solar_absolute.loc["2020-06-01":"2020-06-03"])
plt.show()

adf = adfuller(solar_df)


#%%
# Original Series
fig, axes = plt.subplots(4, 2)
axes[0, 0].plot(solar_df); axes[0, 0].set_title('Original Series')
plot_acf(solar_df, ax=axes[0, 1])

first = solar_df.diff()
# 1st Differencing
axes[1, 0].plot(first); axes[1, 0].set_title('1st Order Differencing')
plot_acf(first.dropna(), ax=axes[1, 1])

second = first.diff()
# 2nd Differencing
axes[2, 0].plot(second); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(second.dropna(), ax=axes[2, 1])

third = second.diff()
# 2nd Differencing
axes[3, 0].plot(third); axes[3, 0].set_title('2nd Order Differencing')
plot_acf(third.dropna(), ax=axes[3, 1])
plt.show()
#%%
train = solar_df[:2190]
test = solar_df[2190:]
model = ARIMA(train, order=(27,1,1))
model_fit = model.fit()
print(model_fit.summary())
model_fit.plot_diagnostics()
plt.show()

# Forecast
fc = model_fit.forecast(24)  # 95% conf
#%%
# Make as pandas series
fc_series = pd.Series(fc, index=test.index)

# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train[2150:], label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='forecast')

plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()

#%%
import numpy as np

def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))
    rmse = np.mean((forecast - actual) ** 2) ** .5
    return rmse

mape = forecast_accuracy(fc, test[:24])
print(mape)

#%%
# Actual vs Fitted
fig, ax = plt.subplots()
ax = solar_df.plot(ax=ax)
fig = pd.DataFrame(model_fit.predict()).plot(ax=ax)
plt.show()

#%%
from pmdarima.arima.utils import ndiffs
y = solar_df
## Adf Test
print(ndiffs(y, test='adf'))  # 2

# KPSS test
print(ndiffs(y, test='kpss'))  # 0

# PP test:
print(ndiffs(y, test='pp'))  # 2
#%%
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(solar_df.diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,5))
plot_pacf(solar_df.diff().dropna(), ax=axes[1])

plt.show()
#%%
# line plot of residuals
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()
# density plot of residuals
residuals.plot(kind='kde')
plt.show()
# summary stats of residuals
print(residuals.describe())

#%%
import pmdarima as pm

smodel = pm.auto_arima(train, start_p=24, start_q=1,
                         test='adf',
                         max_p=27, max_q=3, m=90,
                         start_P=1, seasonal=True,
                         d=None, D=1, trace=True,
                         error_action='ignore',
                         suppress_warnings=True,
                         stepwise=True)

smodel.summary()