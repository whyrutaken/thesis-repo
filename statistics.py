#%%
import pandas as pd
import Printer as mf


df_solar,  = mf.read_csv(["solar-produced.csv"], "2022-02-28")[0]

df_solar = mf.get_new_resolution_by_argument(df_solar, 'H')


# %%
# "daily solar": daily max produced solar

def get_daily_max(table):
    table.index = pd.to_datetime(table.index)
    daily_max = table.loc[table.groupby(pd.Grouper(freq='D')).idxmax().iloc[:, 0]]
    return daily_max

daily_solar = get_daily_max(solar)
# %%
# add y/m/d columns to "daily_solar"
from datetime import datetime

daily_solar['Year'] = [datetime.strftime(i, format="%Y") for i in daily_solar.index]
daily_solar['Month'] = [datetime.strftime(i, format="%m") for i in daily_solar.index]
daily_solar['Day'] = [datetime.strftime(i, format="%d") for i in daily_solar.index]

# %%
# plot daily solar
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style('whitegrid')
plt.figure(figsize=(14, 8))
fig = sns.lineplot(data=daily_solar, x=daily_solar.index, y='Solar energy produced (Wh)', ci=None)
fig.set_title('Daily Power Generation')
plt.show()

# %%
daily_solar = daily_solar.rename(columns={"Solar energy produced (Wh)": "Daily_PV_output"})
# yearly average of daily max solar
daily_solar_2020 = daily_solar.loc['2020-01-01':'2020-12-31']
daily_solar_2021 = daily_solar.loc['2021-01-01':'2021-12-31']
daily_solar_2022 = daily_solar.loc['2022-01-01':'2022-12-31']

# %%
print(daily_solar_2020.Daily_PV_output.mean())
print(daily_solar_2021.Daily_PV_output.mean())
print(daily_solar_2022.Daily_PV_output.mean())


# %%

def entry_update_per_hour(df):
    df["timestamp"] = pd.to_datetime(df.index)
    df = df.set_index("timestamp")
    dates = df.rename_axis('Time').index.floor('H')
    df = df.groupby([dates]).size().reset_index(name='hourly update count')
    return df


solar_entry = entry_update_per_hour(solar)

# %%
electricity["timestamp"] = pd.to_datetime(electricity.index)
electricity = electricity.set_index("timestamp")
dates = electricity.rename_axis('Time').index.floor('24H')
df2 = electricity.groupby([dates]).size().reset_index(name='electricity change count')

# https://stackoverflow.com/questions/53493599/groupby-items-and-count-item-every-hour-in-pandas