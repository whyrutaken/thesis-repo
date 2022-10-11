import pandas as pd
import matplotlib.pyplot as plt


# deprecated
def convert_total_to_daily_accumulation(df: pd.DataFrame) -> pd.DataFrame:
    # BUT modify read_csv() to make it work
    # for rows after 2022-03-01

    new_df = df["2022-03-01":"2022-06-02"]
    new_df = new_df.iloc[::-1]
    new_df = new_df.diff(periods=1).fillna(new_df)
    return new_df


def plot_solar_vs_consumption(df: pd.DataFrame, date: str):
    df[date].solar_absolute.plot(legend=True)
    df[date].consumption_absolute.plot(legend=True)
    plt.show()


def plot_autocorrelation(df: pd.DataFrame, from_date: str, to_date: str):
    x = pd.plotting.autocorrelation_plot(df.loc[from_date:to_date].consumption_absolute)
    x.plot()
    plt.show()


# TODO: instead of deleting completely, you could inject the value from an hour before
def del_lines(df: pd.DataFrame, list_of_dates: list) -> pd.DataFrame:
    for date in list_of_dates:
        df = df[~(df.index == date)]
    return df
