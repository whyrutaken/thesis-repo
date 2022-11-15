#%%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import statistics


def plot_monthly(master_df):
    fig, ax = plt.subplots(2)
    plt.rcParams["figure.figsize"] = (15, 5)
    master_df["2020-01-01 01:00":"2020-01-31"].plot(y='demand_absolute', use_index=True, ax=ax[0], legend=False)
    master_df["2020-02-01":"2020-02-28"].plot(y='demand_absolute', use_index=True, ax=ax[0], legend=False)
    master_df["2020-03-01":"2020-03-31"].plot(y='demand_absolute', use_index=True, ax=ax[0], legend=False)
    master_df["2020-04-01":"2020-04-30"].plot(y='demand_absolute', use_index=True, ax=ax[0], legend=False)
    master_df["2020-05-01":"2020-05-31"].plot(y='demand_absolute', use_index=True, ax=ax[0], legend=False)
    master_df["2020-06-01":"2020-06-30"].plot(y='demand_absolute', use_index=True, ax=ax[0], legend=False)
    master_df["2020-07-01":"2020-07-31"].plot(y='demand_absolute', use_index=True, ax=ax[0], legend=False)
    master_df["2020-08-01":"2020-08-31"].plot(y='demand_absolute', use_index=True, ax=ax[0], legend=False)
    master_df["2020-09-01":"2020-09-30"].plot(y='demand_absolute', use_index=True, ax=ax[0], legend=False)
    master_df["2020-10-01":"2020-10-31"].plot(y='demand_absolute', use_index=True, ax=ax[0], legend=False)
    master_df["2020-11-01":"2020-11-30"].plot(y='demand_absolute', use_index=True, ax=ax[0], legend=False)
    master_df["2020-12-01":"2020-12-31"].plot(y='demand_absolute', use_index=True, ax=ax[0], legend=False)

    master_df["2021-01-01":"2021-01-31"].plot(y='demand_absolute', use_index=True, ax=ax[1], legend=False)
    master_df["2021-02-01":"2021-02-28"].plot(y='demand_absolute', use_index=True, ax=ax[1], legend=False)
    master_df["2021-03-01":"2021-03-31"].plot(y='demand_absolute', use_index=True, ax=ax[1], legend=False)
    master_df["2021-04-01":"2021-04-30"].plot(y='demand_absolute', use_index=True, ax=ax[1], legend=False)
    master_df["2021-05-01":"2021-05-31"].plot(y='demand_absolute', use_index=True, ax=ax[1], legend=False)
    master_df["2021-06-01":"2021-06-30"].plot(y='demand_absolute', use_index=True, ax=ax[1], legend=False)
    master_df["2021-07-01":"2021-07-31"].plot(y='demand_absolute', use_index=True, ax=ax[1], legend=False)
    master_df["2021-08-01":"2021-08-31"].plot(y='demand_absolute', use_index=True, ax=ax[1], legend=False)
    master_df["2021-09-01":"2021-09-30"].plot(y='demand_absolute', use_index=True, ax=ax[1], legend=False)
    master_df["2021-10-01":"2021-10-31"].plot(y='demand_absolute', use_index=True, ax=ax[1], legend=False)
    master_df["2021-11-01":"2021-11-30"].plot(y='demand_absolute', use_index=True, ax=ax[1], legend=False)
    master_df["2021-12-01":"2021-12-31"].plot(y='demand_absolute', use_index=True, ax=ax[1], legend=False)

    plt.show()


def plot_years_seasonally(master_df, attr, y_label):
    fig, ax = plt.subplots(2)
    fig.tight_layout(pad=3.0)
    plt.rcParams["figure.figsize"] = (15, 6)
    master_df["2020-01-01":"2020-02-28"].plot(y=attr, use_index=True, ax=ax[0], legend=False, color="royalblue")
    master_df["2020-03-01":"2020-05-31"].plot(y=attr, use_index=True, ax=ax[0], legend=False, color="lightpink")
    master_df["2020-06-01":"2020-08-31"].plot(y=attr, use_index=True, ax=ax[0], legend=False, color="limegreen")
    master_df["2020-09-01":"2020-11-30"].plot(y=attr, use_index=True, ax=ax[0], legend=False, color="darkorange")
    master_df["2020-12-01":"2020-12-31"].plot(y=attr, use_index=True, ax=ax[0], legend=False, color="royalblue")


    master_df["2021-01-01":"2021-02-28"].plot(y=attr, use_index=True, ax=ax[1], legend=False, color="royalblue")
    master_df["2021-03-01":"2021-05-31"].plot(y=attr, use_index=True, ax=ax[1], legend=False, color="lightpink")
    master_df["2021-06-01":"2021-08-31"].plot(y=attr, use_index=True, ax=ax[1], legend=False, color="limegreen")
    master_df["2021-09-01":"2021-11-30"].plot(y=attr, use_index=True, ax=ax[1], legend=False, color="darkorange")
    master_df["2021-12-01":"2021-12-31"].plot(y=attr, use_index=True, ax=ax[1], legend=False, color="royalblue")


    plt.setp(ax[:], xlabel='Time')
    plt.setp(ax[:], ylabel=y_label)
    plt.show()


def plot_weeks_seasonally(master_df, attr, y_label):
    fig, ax = plt.subplots(4)
    fig.tight_layout(pad=3.0)
    plt.rcParams["figure.figsize"] = (15, 7)
 #   x_ticks = list(master_df["2021-01-11":"2021-01-17"].index.floor('D').drop_duplicates())
 #   plt.axvline(x=x_ticks, color='b', ax=ax[0])

    master_df["2021-01-11":"2021-01-17"].plot(y=attr, use_index=True, ax=ax[0], legend=False, color="royalblue")
    master_df["2021-04-12":"2021-04-18"].plot(y=attr, use_index=True, ax=ax[1], legend=False, color="lightpink")
    master_df["2021-07-05":"2021-07-11"].plot(y=attr, use_index=True, ax=ax[2], legend=False, color="limegreen")
    master_df["2021-10-04":"2021-10-10"].plot(y=attr, use_index=True, ax=ax[3], legend=False, color="darkorange")

 #   master_df["2021-08-02":"2021-08-08"].plot(y=attr, use_index=True, ax=ax[1], legend=False, color="royalblue")
  #  master_df["2021-03-01":"2021-05-31"].plot(y=attr, use_index=True, ax=ax[1], legend=False, color="lightpink")
  #  master_df["2021-06-01":"2021-08-31"].plot(y=attr, use_index=True, ax=ax[1], legend=False, color="limegreen")
  #  master_df["2021-09-01":"2021-11-30"].plot(y=attr, use_index=True, ax=ax[1], legend=False, color="darkorange")
   # master_df["2021-12-01":"2021-12-31"].plot(y=attr, use_index=True, ax=ax[1], legend=False, color="royalblue")


    plt.setp(ax[:], xlabel='Time')
    plt.setp(ax[:], ylabel=y_label)
    plt.show()

master_df = pd.read_csv("extracted-data/master-df.csv")
master_df.loc["2020-01-01 00:00:00+00:00"] = master_df.loc["2020-01-01 01:00:00+00:00"]
index = master_df.index.sort_values()
master_df.index = pd.DatetimeIndex(index)

stat = statistics.Statistics()
master_df = stat.get_season(master_df)
plot_monthly(master_df)
plot_years_seasonally(master_df, 'solar_absolute', "PV production [Wh]")
plot_weeks_seasonally(master_df, "demand_absolute", "Demand [Wh]")
plot_weeks_seasonally(master_df, "solar_absolute", "PV production [Wh]")

