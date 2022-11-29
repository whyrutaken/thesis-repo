
import pandas as pd
from preparator import Preparator
from svr_model import SVRModel
from arima_model import ArimaModel
import printer
import matplotlib.pyplot as plt
#%%
if __name__ == '__main__':
    svr = SVRModel("solar_absolute", test_from_date="2020-06-10 00:00", test_to_date="2020-06-12 00:00", horizon=24)
    arima = ArimaModel("solar_absolute", test_from_date="2020-06-10 00:00", test_to_date="2020-06-12 00:00", forecast_steps=24)

    #%%
    import matplotlib.pyplot as plt


    def print_forecast(train, test, pred1, pred2):
        fig, ax = plt.subplots(figsize=(15, 5), dpi=100)
        plt.locator_params(axis='x', nbins=5)
        ax.plot(train[len(train) - 48:], label='training')
        ax.plot(test[:168], label='actual', color="lightcoral")
        ax.plot(pred1, label='SVR forecast 24h', color="darkgreen")
        ax.plot(pred2, label='ARIMA forecast 24h', color="gold")
        #    plt.plot(pred3, label='forecast 48h')
        ax.legend()
        ax.set(xlabel="Time", ylabel="PV production [Wh]", title="Forecast vs Actual")
        ax.grid(True, which='both')
        plt.show()


    print_forecast(svr.y_train, svr.y_test, svr.prediction, arima.prediction)
