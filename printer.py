import pandas as pd
import matplotlib.pyplot as plt

def print_forecast(train, test, pred1, pred2):
    fig, ax = plt.subplots(figsize=(15, 5), dpi=100)
    plt.locator_params(axis='x', nbins=5)
    ax.plot(train[len(train) - 48:], label='training')
    ax.plot(test[:240], label='actual')
    ax.plot(pred1, label='svr forecast 10h')
    ax.plot(pred2, label='arima forecast 5h')
#    plt.plot(pred3, label='forecast 48h')
    ax.set(xlabel="Time", ylabel="PV production [Wh]", title="Forecast vs Actual")
    ax.grid(True, which='both')
    plt.show()