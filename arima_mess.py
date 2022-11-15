#%%
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot as plt
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
import preprocessor

#["2020-06-01":"2020-09-03"]
#%%
class ARIMAModel:
    def __init__(self, attribute, from_date, to_date):
        self.df = preprocessor.Preprocessor().get_master_df(attribute, from_date, to_date)
    #    self.print_autocorrelation_plot("2020-06-03", "2020-06-06")
        self.df = self.difference(self.df, 365*24)
        self.train_df, self.test_df = self.train_test_split(self.df)
        self.print_series_vs_autocorrelation(self.train_df, self.train_df)
        self.fitted_model = self.fit_model(self.train_df)
        self.forecast = self.forecast(self.train_df, self.fitted_model)

        inv_forecast = self.inverse_difference(self.df, self.forecast, self.df["2020-01-02"].get_loc(), 365*24)

        self.print_forecast(24)


    # source: https://machinelearningmastery.com/make-sample-forecasts-arima-python/
    def difference(self, df, interval=1):
        diff = list()
        for i in range(interval, len(df)):
            value = df[i] - df[i - interval]
            diff.append(value)
        return np.array(diff)

    def inverse_difference(self, history, yhat, position, interval=1):
        return yhat + history[position-interval]

    def train_test_split(self, df):
        limit = "2022-01-01"
        train = pd.Series(df[:limit], index=df[:limit].index)
        test = pd.Series(df[limit:], index=df[limit:].index)
        return train, test

    def print_autocorrelation_plot(self, from_date, to_date):
        autocorrelation_plot(self.df.loc[from_date:to_date])
        plt.show()

    def print_series_vs_autocorrelation(self, df, diff_df):
        fig, axes = plt.subplots(2, 2)
        axes[0, 0].plot(df); axes[0, 0].set_title('Original Series')
        plot_acf(df, ax=axes[0, 1])

        axes[1, 0].plot(diff_df); axes[1, 0].set_title('Differenced Series')
        plot_acf(diff_df, ax=axes[1, 1])

        plt.show()

    def fit_model(self, train):
        model = ARIMA(train, order=(24, 1, 1))
        fitted = model.fit()
        fitted.plot_diagnostics()
        plt.show()
        return fitted

    def forecast(self, train_df, fitted_model):
        forecast = fitted_model.prediction(1)
        forecast = self.inverse_difference(train_df, forecast, 365*24)
        print('Forecast: %f' % forecast)
        return forecast


    def print_forecast(self, forecast_steps):
        fc = self.fitted_model.forecast(forecast_steps)  # 95% conf

        fc_series = pd.Series(fc, index=self.test_df.index)

        plt.figure(figsize=(12, 5), dpi=100)
        plt.plot(self.train_df[1900:], label='training')
        plt.plot(self.test_df[:72], label='actual')
        plt.plot(fc_series[:72], label='forecast')

        plt.title('Forecast vs Actuals')
        plt.legend(loc='upper left', fontsize=8)
        plt.show()

    def forecast_accuracy(self, forecast, actual):
        mape = np.mean(np.abs(forecast - actual) / np.abs(actual))
        rmse = np.mean((forecast - actual) ** 2) ** .5
        mae = np.mean(np.abs(forecast - actual))
        return mae

model_ = ARIMAModel("solar_absolute", "2020-01-01", "2022-02-28")
#%%
master_df = pd.read_csv("extracted-data/master-df.csv")
solar_df = master_df.solar_absolute["2020-06-02":"2020-12-04"]
solar_df = solar_df.reset_index(drop=True)
load_df = master_df.demand_absolute["2021-01-01":"2022-01-01"]
autocorrelation_plot(master_df.solar_absolute.loc["2020-06-01":"2020-06-03"])
plt.show()

adf = adfuller(solar_df)


#%%
train = solar_df[:2190]
test = solar_df[2190:]
model = ARIMA(train, order=(27,1,1))
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


#%%

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
import numpy as np

def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))
    rmse = np.mean((forecast - actual) ** 2) ** .5
    mae = np.mean(np.abs(forecast - actual))
    return mae

mape = forecast_accuracy(fc, test[:24])
print(mape)

#%%


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

