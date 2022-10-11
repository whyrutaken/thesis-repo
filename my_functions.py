import pandas as pd
import matplotlib.pyplot as plt





def get_new_resolution_by_argument(df: pd.DataFrame, resolution: str) -> pd.DataFrame:
    resampled_df = df.resample(resolution).max()
    return resampled_df.fillna(value=0)











# deprecated
def convert_total_to_daily_accumulation(df: pd.DataFrame) -> pd.DataFrame:
    # BUT modify read_csv() to make it work
    # for rows after 2022-03-01

    new_df = df["2022-03-01":"2022-06-02"]
    new_df = new_df.iloc[::-1]
    new_df = new_df.diff(periods=1).fillna(new_df)
    return new_df


def get_abs_value_from_daily_acc(df: pd.DataFrame, old_column: str, new_column: str) -> pd.DataFrame:
    df[new_column] = df[old_column].diff().fillna(0)  # get the difference of elements from previous elements

    # get date change locations where diff != 0 Days
    # (in df2["add"], True: whenever the day changed, False: whenever is a value from the same day)
    df["date"] = df.index.date
    df["add"] = df["date"].diff().ne("0D")
    # 3. add the previous total back
    df.loc[df["add"], new_column] += df[old_column].shift()[df["add"]]
    df = df.fillna(0)
    del df["date"]
    del df["add"]
    return df


def calculate_consumption(imported: float, exported: float, solar: float) -> float:
    used = imported + solar
    return used - exported


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
