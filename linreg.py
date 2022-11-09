#%%
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.nonparametric.smoothers_lowess import lowess
import numpy as np
import scipy.stats as stats

master_df = pd.read_csv("extracted-data/master-df.csv")
weather_df = pd.read_csv("historical-weather.csv")
#weather_df.dt_ise = pd.to_datetime(weather_df.dt_iso).strftime('%Y-%m-%dT%H:%M:%SZ')
weather_df = weather_df.set_index("dt_iso")

master_df = master_df[:2400]
weather_df = weather_df[:2400]

weather_df = weather_df.reset_index()
master_df = master_df.reset_index()

X = weather_df.temp
X_ = sm.add_constant(weather_df.temp)
Y = master_df.solar_absolute
model = sm.OLS(Y, X_)
results = model.fit()
print(results.summary())

plt.scatter(X, Y, alpha=0.3)
y_predict = results.params[0] + results.params[1]*X
plt.plot(X, y_predict, linewidth=3)

plt.xlabel('Temperature')
plt.ylabel('Solar')
plt.title('OLS Regression')
plt.show()

