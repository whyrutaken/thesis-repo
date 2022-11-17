#%%
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
import seaborn as sns
import statistics
import numpy as np


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


def plot_year_and_season(df, variable, legend):
    font = {'family': 'normal',
            'size': 15}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(nrows=2, ncols=1)
    fig.set_size_inches(20, 8)
    df["2020-01-01":"2020-02-28"].plot(y=variable, use_index=True, ax=ax[0], legend=False, color="royalblue", alpha=0.9)
    df["2020-03-01":"2020-05-31"].plot(y=variable, use_index=True, ax=ax[0], legend=False, color="lightpink", alpha=0.9)
    df["2020-06-01":"2020-08-31"].plot(y=variable, use_index=True, ax=ax[0], legend=False, color="limegreen", alpha=0.9)
    df["2020-09-01":"2020-11-30"].plot(y=variable, use_index=True, ax=ax[0], legend=False, color="darkorange", alpha=0.9)
    df["2020-12-01":"2020-12-31"].plot(y=variable, use_index=True, ax=ax[0], legend=False, color="royalblue", alpha=0.9)


    df["2021-01-01":"2021-02-28"].plot(y=variable, use_index=True, ax=ax[1], legend=False, color="royalblue", alpha=0.9)
    df["2021-03-01":"2021-05-31"].plot(y=variable, use_index=True, ax=ax[1], legend=False, color="lightpink", alpha=0.9)
    df["2021-06-01":"2021-08-31"].plot(y=variable, use_index=True, ax=ax[1], legend=False, color="limegreen", alpha=0.9)
    df["2021-09-01":"2021-11-30"].plot(y=variable, use_index=True, ax=ax[1], legend=False, color="darkorange", alpha=0.9)
    df["2021-12-01":"2021-12-31"].plot(y=variable, use_index=True, ax=ax[1], legend=False, color="royalblue", alpha=0.9)

    ax[0].set_ylim(bottom=0)
    ax[1].set_ylim(bottom=0)
    if variable == "demand_absolute":
        ax[0].set_ylim(top=50000)
        ax[1].set_ylim(top=50000)
        ax[0].legend(legend, bbox_to_anchor=(0.9, 0.8))
        ax[1].legend(legend, bbox_to_anchor=(0.9, 0.8))
        fig.set_size_inches(24, 10)

    if variable == "solar_absolute":
        ax[0].set_ylim(top=25000)
        ax[1].set_ylim(top=25000)
        ax[0].legend(legend)
        ax[1].legend(legend)

    if variable == "imported_absolute":
        ax[0].set_ylim(top=40000)
        ax[1].set_ylim(top=40000)
        ax[0].legend(legend, bbox_to_anchor=(0.9, 0.8))
        ax[1].legend(legend, bbox_to_anchor=(0.9, 0.8))
        fig.set_size_inches(24, 10)


    if variable == "exported_absolute":
        ax[0].set_ylim(top=20000)
        ax[1].set_ylim(top=20000)
        ax[0].legend(legend)
        ax[1].legend(legend)

    ax[0].vlines(["2020-03-01", "2020-06-01", "2020-09-01", "2020-12-01"], ymin=0, ymax=50000, linestyles="dashed", colors='black')
    ax[1].vlines(["2021-03-01", "2021-06-01", "2021-09-01", "2021-12-01"], ymin=0, ymax=50000, linestyles="dashed", colors='black')



    ax[0].title.set_text("2020")
    ax[1].title.set_text("2021")

    fig.tight_layout(pad=1.5)
    for ax_ in range(len(ax)):
        ax[ax_].grid(True, which='both')
    plt.setp(ax[:], ylabel="Electricity [Wh]")
    plt.show()


def plot_cons_vs_prod_per_year(df):
    font = {'family': 'normal',
            'size': 15}
    matplotlib.rc('font', **font)
    fig, ax = plt.subplots(nrows=2, ncols=1)
    fig.set_size_inches(20, 8)
    df["2020-01-01":"2020-02-28"].plot(y="demand_absolute", use_index=True, ax=ax[0], legend=False, color="royalblue", alpha=0.9)
    df["2020-01-01":"2020-02-28"].plot(y="solar_absolute", use_index=True, ax=ax[0], legend=False, color="gold", alpha=0.8)

    df["2020-03-01":"2020-05-31"].plot(y="demand_absolute", use_index=True, ax=ax[0], legend=False, color="royalblue", alpha=0.9)
    df["2020-03-01":"2020-05-31"].plot(y="solar_absolute", use_index=True, ax=ax[0], legend=False, color="gold", alpha=0.8)

    df["2020-06-01":"2020-08-31"].plot(y="demand_absolute", use_index=True, ax=ax[0], legend=False, color="royalblue", alpha=0.9)
    df["2020-06-01":"2020-08-31"].plot(y="solar_absolute", use_index=True, ax=ax[0], legend=False, color="gold", alpha=0.8)

    df["2020-09-01":"2020-11-30"].plot(y="demand_absolute", use_index=True, ax=ax[0], legend=False, color="royalblue", alpha=0.9)
    df["2020-09-01":"2020-11-30"].plot(y="solar_absolute", use_index=True, ax=ax[0], legend=False, color="gold", alpha=0.8)

    df["2020-12-01":"2020-12-31"].plot(y="demand_absolute", use_index=True, ax=ax[0], legend=False, color="royalblue", alpha=0.9)
    df["2020-12-01":"2020-12-31"].plot(y="solar_absolute", use_index=True, ax=ax[0], legend=False, color="gold", alpha=0.8)


    df["2021-01-01":"2021-02-28"].plot(y="demand_absolute", use_index=True, ax=ax[1], legend=False, color="royalblue", alpha=0.9)
    df["2021-01-01":"2021-02-28"].plot(y="solar_absolute", use_index=True, ax=ax[1], legend=False, color="gold", alpha=0.8)

    df["2021-03-01":"2021-05-31"].plot(y="demand_absolute", use_index=True, ax=ax[1], legend=False, color="royalblue", alpha=0.9)
    df["2021-03-01":"2021-05-31"].plot(y="solar_absolute", use_index=True, ax=ax[1], legend=False, color="gold", alpha=0.8)

    df["2021-06-01":"2021-08-31"].plot(y="demand_absolute", use_index=True, ax=ax[1], legend=False, color="royalblue", alpha=0.9)
    df["2021-06-01":"2021-08-31"].plot(y="solar_absolute", use_index=True, ax=ax[1], legend=False, color="gold", alpha=0.8)

    df["2021-09-01":"2021-11-30"].plot(y="demand_absolute", use_index=True, ax=ax[1], legend=False, color="royalblue", alpha=0.9)
    df["2021-09-01":"2021-11-30"].plot(y="solar_absolute", use_index=True, ax=ax[1], legend=False, color="gold", alpha=0.8)

    df["2021-12-01":"2021-12-31"].plot(y="demand_absolute", use_index=True, ax=ax[1], legend=False, color="royalblue", alpha=0.9)
    df["2021-12-01":"2021-12-31"].plot(y="solar_absolute", use_index=True, ax=ax[1], legend=False, color="gold", alpha=0.8)

    ax[0].set_ylim(bottom=0)
    ax[1].set_ylim(bottom=0)
    ax[0].set_ylim(top=50000)
    ax[1].set_ylim(top=50000)

    ax[0].vlines(["2020-03-01", "2020-06-01", "2020-09-01", "2020-12-01"], ymin=0, ymax=50000, linestyles="dashed", colors='black')
    ax[1].vlines(["2021-03-01", "2021-06-01", "2021-09-01", "2021-12-01"], ymin=0, ymax=50000, linestyles="dashed", colors='black')

    ax[0].legend(["Power consumption", "PV production"])
    ax[1].legend(["Power consumption", "PV production"])

    ax[0].title.set_text("2020")
    ax[1].title.set_text("2021")

    fig.tight_layout(pad=1.5)
    for ax_ in range(len(ax)):
        ax[ax_].grid(True, which='both')
    plt.setp(ax[:], ylabel="Electricity [Wh]")
    plt.show()





def plot_cons_vs_prod_displaying_weeks_of_seasons(df):
    font = {'family': 'normal',
            'size': 15}
    matplotlib.rc('font', **font)

    fig, ax = plt.subplots(nrows=4, ncols=1)
    fig.set_size_inches(17, 14)
    ax[0] = plt.subplot2grid((4,3),(0,0),colspan=3)
    ax[1] = plt.subplot2grid((4,3),(1,0),colspan=3)
    ax[2] = plt.subplot2grid((4,3),(2,0),colspan=3)
    ax[3] = plt.subplot2grid((4,3),(3,0),colspan=3)

    plt.setp(ax[:], ylabel="Electricity [Wh]")


    df["2020-01-13":"2020-01-19"].plot(y="demand_absolute", use_index=True, ax=ax[0], legend=False, color="royalblue")
    df["2020-01-13":"2020-01-19"].plot(y="solar_absolute", use_index=True, ax=ax[0], legend=False, color="gold")

    df["2020-05-11":"2020-05-17"].plot(y="demand_absolute", use_index=True, ax=ax[1], legend=False, color="lightpink")
    df["2020-05-11":"2020-05-17"].plot(y="solar_absolute", use_index=True, ax=ax[1], legend=False, color="gold")

    df["2020-07-06":"2020-07-12"].plot(y="demand_absolute", use_index=True, ax=ax[2], legend=False, color="limegreen")
    df["2020-07-06":"2020-07-12"].plot(y="solar_absolute", use_index=True, ax=ax[2], legend=False, color="gold")

    df["2020-10-12":"2020-10-18"].plot(y="demand_absolute", use_index=True, ax=ax[3], legend=False, color="darkorange")
    df["2020-10-12":"2020-10-18"].plot(y="solar_absolute", use_index=True, ax=ax[3], legend=False, color="gold")


    ax[0].set_ylim(bottom=0)
    ax[1].set_ylim(bottom=0)
    ax[2].set_ylim(bottom=0)
    ax[3].set_ylim(bottom=0)
    ax[0].set_ylim(top=40000)
    ax[1].set_ylim(top=40000)
    ax[2].set_ylim(top=40000)
    ax[3].set_ylim(top=40000)

    ax[0].legend(["Power consumption", "PV production"])
    ax[1].legend(["Power consumption", "PV production"])
    ax[2].legend(["Power consumption", "PV production"])
    ax[3].legend(["Power consumption", "PV production"])


    ax[0].title.set_text("Winter")
    ax[1].title.set_text("Spring")
    ax[2].title.set_text("Summer")
    ax[3].title.set_text("Autumn")

    fig.tight_layout()
    for ax_ in range(len(ax)):
        ax[ax_].grid(True, which='both')
    plt.show()


def plot_cons_vs_prod_displaying_months_of_seasons(df):
    font = {'family': 'normal',
            'size': 15}
    matplotlib.rc('font', **font)

    fig, ax = plt.subplots(nrows=4, ncols=1)
    fig.set_size_inches(17, 14)
    ax[0] = plt.subplot2grid((4,3),(0,0),colspan=3)
    ax[1] = plt.subplot2grid((4,3),(1,0),colspan=3)
    ax[2] = plt.subplot2grid((4,3),(2,0),colspan=3)
    ax[3] = plt.subplot2grid((4,3),(3,0),colspan=3)

    plt.setp(ax[:], ylabel="Electricity [Wh]")

    df["2020-01-01":"2020-01-31"].plot(y="demand_absolute", use_index=True, ax=ax[0], legend=False, color="royalblue")
    df["2020-01-01":"2020-01-31"].plot(y="solar_absolute", use_index=True, ax=ax[0], legend=False, color="gold")

    df["2020-05-01":"2020-05-31"].plot(y="demand_absolute", use_index=True, ax=ax[1], legend=False, color="lightpink")
    df["2020-05-01":"2020-05-31"].plot(y="solar_absolute", use_index=True, ax=ax[1], legend=False, color="gold")

    df["2020-07-01":"2020-07-31"].plot(y="demand_absolute", use_index=True, ax=ax[2], legend=False, color="limegreen")
    df["2020-07-01":"2020-07-31"].plot(y="solar_absolute", use_index=True, ax=ax[2], legend=False, color="gold")

    df["2020-10-01":"2020-10-31"].plot(y="demand_absolute", use_index=True, ax=ax[3], legend=False, color="darkorange")
    df["2020-10-01":"2020-10-31"].plot(y="solar_absolute", use_index=True, ax=ax[3], legend=False, color="gold")


    ax[0].set_ylim(bottom=0)
    ax[1].set_ylim(bottom=0)
    ax[2].set_ylim(bottom=0)
    ax[3].set_ylim(bottom=0)
    ax[0].set_ylim(top=40000)
    ax[1].set_ylim(top=40000)
    ax[2].set_ylim(top=40000)
    ax[3].set_ylim(top=40000)

    ax[0].vlines(["2020-01-06", "2020-01-13", "2020-01-20", "2020-01-27"], ymin=0, ymax=40000, linestyles="dashed",
                 colors='black')
    ax[1].vlines(["2020-05-04", "2020-05-11", "2020-05-18", "2020-05-25"], ymin=0, ymax=40000, linestyles="dashed",
                 colors='black')
    ax[2].vlines(["2020-07-06", "2020-07-13", "2020-07-20", "2020-07-27"], ymin=0, ymax=40000, linestyles="dashed",
                 colors='black')
    ax[3].vlines(["2020-10-05", "2020-10-12", "2020-10-19", "2020-10-26"], ymin=0, ymax=40000, linestyles="dashed",
                 colors='black')

    ax[0].legend(["Power consumption", "PV production"])
    ax[1].legend(["Power consumption", "PV production"])
    ax[2].legend(["Power consumption", "PV production"])
    ax[3].legend(["Power consumption", "PV production"])


    ax[0].title.set_text("Winter")
    ax[1].title.set_text("Spring")
    ax[2].title.set_text("Summer")
    ax[3].title.set_text("Autumn")

    fig.tight_layout()
    for ax_ in range(len(ax)):
        ax[ax_].grid(True, which='both')
    plt.show()



master_df = pd.read_csv("extracted-data/master-df.csv")
master_df.loc["2020-01-01 00:00:00+00:00"] = master_df.loc["2020-01-01 01:00:00+00:00"]
index = master_df.index.sort_values()
master_df.index = pd.DatetimeIndex(index)

stat = statistics.Statistics()
master_df = stat.get_season(master_df)


plot_cons_vs_prod_per_year(master_df)
plot_year_and_season(master_df, "solar_absolute", ["Winter PV production", "Spring PV production", "Summer PV production", "Autumn PV production"])
plot_year_and_season(master_df, "demand_absolute", ["Winter power consumption", "Spring power consumption", "Summer power consumption", "Autumn power consumption"])

plot_year_and_season(master_df, "imported_absolute", ["Winter imported power", "Spring imported power", "Summer imported power", "Autumn imported power"])
plot_year_and_season(master_df, "exported_absolute", ["Winter exported power", "Spring exported power", "Summer exported power", "Autumn exported power"])


plot_cons_vs_prod_displaying_months_of_seasons(master_df)
plot_cons_vs_prod_displaying_weeks_of_seasons(master_df)


plt.show()

