import pandas as pd
import matplotlib.pyplot as plt


def print_single_forecast(train, test, pred1):
    fig, ax = plt.subplots(figsize=(15, 5), dpi=100)
    plt.locator_params(axis='x', nbins=5)
    ax.plot(train[len(train) - 48:], label='training')
    ax.plot(test[:168], label='actual', color="lightcoral")

    ax.plot(pred1[:168], label='LSTM forecast 24h', color="gold")
    ax.legend()
    ax.set(xlabel="Time", ylabel="PV production [Wh]", title="Forecast vs Actual")
    ax.grid(True, which='both')
    plt.show()

def print_double_forecast(train, test, pred1, pred2):
    fig, ax = plt.subplots(figsize=(15, 5), dpi=100)
    plt.locator_params(axis='x', nbins=5)
    ax.plot(train[len(train) - 48:], label='training')
    ax.plot(test[:168], label='actual', color="lightcoral")

    ax.plot(pred1[:168], label='LSTM forecast 24h', color="gold")
    ax.plot(pred2[:168], label='SVR forecast 24h', color="darkgreen")

    ax.legend()
    ax.set(xlabel="Time", ylabel="PV production [Wh]", title="Forecast vs Actual")
    ax.grid(True, which='both')
    plt.show()

def print_multi_forecast(train, test, pred1, pred2, pred3):
    fig, ax = plt.subplots(figsize=(15, 5), dpi=100)
    plt.locator_params(axis='x', nbins=5)
    ax.plot(train[len(train) - 48:], label='training')
    ax.plot(test[:48], label='actual', color="tomato")
    ax.plot(pred1, label='SVR forecast 24h', color="darkgreen")
    ax.plot(pred2, label='ARIMA forecast 24h', color="gold")
    ax.plot(pred3, label='LSTM forecast 24h', color="darkgoldenrod")
    ax.legend()
    ax.set(xlabel="Time", ylabel="PV production [Wh]", title="Forecast vs Actual")
    ax.grid(True, which='both')
    plt.show()

def print_error(error):
    fig, ax = plt.subplots(figsize=(15, 5), dpi=100)
    plt.locator_params(axis='x', nbins=5)
    ax.plot(error, label='error')

    ax.set(xlabel="Time", ylabel="RMSE [Wh]", title="Error plot")
    ax.grid(True, which='both')
    plt.show()