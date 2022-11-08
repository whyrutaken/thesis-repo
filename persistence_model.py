#%%
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error


master_df = pd.read_csv("extracted-data/master-df.csv")
weather_df = pd.read_csv("historical-weather.csv")

master_df["2020-01-01":"2020-01-07"].solar_absolute.plot()
plt.show()

#%%

# * Persistence model *

# Create lagged dataset
values = pd.DataFrame(master_df.solar_absolute["2020-01-01":"2020-01-05"])
dataframe = pd.concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
print(dataframe.head(5))

# split into train and test sets
X = dataframe.values
train_size = int(len(X) * 0.66)
train, test = X[1:train_size], X[train_size:]
train_X, train_y = train[:, 0], train[:, 1]
test_X, test_y = test[:, 0], test[:, 1]


# persistence model
def model_persistence(x):
    return x


# walk-forward validation
predictions = list()
for x in test_X:
    yhat = model_persistence(x)
    predictions.append(yhat)
test_score = mean_squared_error(test_y, predictions)
print('Test MSE: %.3f' % test_score)

# plot predictions and expected results
plt.plot(train_y)
plt.plot([None for i in train_y] + [x for x in test_y], color="green")
plt.plot([None for i in train_y] + [x for x in predictions], color="red", label="prediction")

plt.show()