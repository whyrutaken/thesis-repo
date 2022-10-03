# %%
import pandas as pd


def get_hourly_max(table):
    table.index = pd.to_datetime(table.index)
    hourly_max = table.resample('H').max()
    return hourly_max


def fill_nan_with_zero(table):
    new_table = table.fillna(value=0)
    return new_table


solar_data = pd.read_csv("solar-produced.csv", index_col=0)
hourly_solar = get_hourly_max(solar_data)
hourly_solar = fill_nan_with_zero(hourly_solar)


