import pandas as pd
import matplotlib.pyplot as plt





def plot_solar_vs_consumption(df: pd.DataFrame, date: str):
    df[date].solar_absolute.plot(legend=True)
    df[date].consumption_absolute.plot(legend=True)
    plt.show()


def plot_autocorrelation(df: pd.DataFrame, from_date: str, to_date: str):
    x = pd.plotting.autocorrelation_plot(df.loc[from_date:to_date].consumption_absolute)
    x.plot()
    plt.show()


