#%%
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot as plt
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
import preprocessor


master_df = pd.read_csv("extracted-data/master-df.csv")
solar_df = master_df.solar_absolute["2020-01-01":"2021-01-01"]
solar_df = solar_df.reset_index(drop=True)
load_df = master_df.demand_absolute["2020-01-01":"2021-01-01"]
load_df = load_df.reset_index(drop=True)
autocorrelation_plot(master_df.solar_absolute.loc["2020-06-01":"2020-06-03"])
plt.show()

adf = adfuller(solar_df)
adf_load = adfuller(load_df)




#%%
df = solar_df
fig, axes = plt.subplots(3, 2, figsize=(10, 10))
axes[0, 0].plot(df); axes[0, 0].set_title('Original Series')
plot_acf(df, ax=axes[0, 1])


first = df.diff()
# 1st Differencing
axes[1, 0].plot(first); axes[1, 0].set_title('1st Order Differencing')
plot_acf(first.dropna(), ax=axes[1, 1])
adf_first = adfuller(first.dropna())


second = first.diff()
# 2nd Differencing
axes[2, 0].plot(second); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(second.dropna(), ax=axes[2, 1])
plt.show()
adf_second = adfuller(second.dropna())

#%%
third = second.diff()
# 2nd Differencing
axes[3, 0].plot(third); axes[3, 0].set_title('2nd Order Differencing')
plot_acf(third.dropna(), ax=axes[3, 1])
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

df = solar_df
df = df.diff().dropna()

fig, axes = plt.subplots(1, 2)
axes[0].plot(df); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,1))
plot_pacf(df, ax=axes[1], lags=40)

plt.show()

#%%
train = solar_df[:2190]
test = solar_df[2190:]
model = ARIMA(train, order=(24,1,1))
model_fit = model.fit()
print(model_fit.summary())
model_fit.plot_diagnostics()
plt.show()

# Forecast
fc = model_fit.forecast(1)  # 95% conf

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

# line plot of residuals
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()
# density plot of residuals
residuals.plot(kind='kde')
plt.show()
# summary stats of residuals
print(residuals.describe())

