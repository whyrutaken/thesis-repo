# %%
import pandas as pd

# import json
id_dictionary = {
    "af8313d0-22c2-49bc-a736-8cde9d6b6415": "Heat 0th floor (kWh)",
    "e28aa982-3ab8-4be2-9954-8c0ee699dceb": "Water 0th floor (kWh)",
    "5c10d36c-99e6-49b5-af06-0df5b9985b71": "Heat 1st floor (kWh)",
    "5419cd17-8f71-4139-afa1-dec0bbfd21a4": "Water 1st floor (kWh)",
    "481b48aa-5bba-44ae-a5ab-2c78c68bfcc5": "Heat 2nd floor (kWh)",
    "00a46798-af26-49cd-84e9-fd64a69cbe24": "Water 2nd floor (kWh)",
    "70ccaf33-1575-4bd5-ae4b-3cb6d857731a": "Electricity (kWh)",
    "bbc8c07b-b8e0-47e7-a4f3-7fb8d4768260": "Solar energy produced (Wh)",
    "b007fc66-9715-4e5a-b4dc-1540c92de99e": "Power imported from Grid (Wh)",
    "40299d14-bf30-4747-8f72-edfe7c26c15a": "Power exported to Grid (Wh)",
    "74111f93-f893-46f7-b500-5fcaf74a16d9": "CO2 0th",
    "eefcd3a0-272d-4090-996d-ecbd99137956": "Temp 0th",
    "f69e449c-3276-41a7-bf1d-d5c0e3f5cae6": "CO2 0th",
    "da0df84f-8455-44fc-b21f-bb66f0dab59a": "Temp 0th",
    "e5d40546-2e08-49af-a457-c1f7377739db": "CO2 1st",
    "65f7b721-0cd8-45fb-9d19-695ffbf433a7": "Temp 1st",
    "7f662fa2-3b32-4981-aee8-796d211038b4": "CO2 1st",
    "f350f38b-548e-4b41-a225-447ab9927aa5": "Temp 1st",
    "ebd40abc-eafe-4ad5-aa2a-dc7eec277515": "CO2 1st",
    "aad2b139-67c8-417f-863b-b07eb083fada": "Temp 1st",
    "23ac2a3d-1fe1-4de5-a89d-c9f25031e00b": "CO2 1st",
    "9c4403ae-a00e-48c9-9467-068aad70fe19": "Temp 1st",
    "14afe6b9-dbfe-4e7f-b0af-cd0d03c72311": "CO2 1st",
    "83e47ae6-53e7-4be3-8054-825d9decd02a": "Temp 1st",
    "4c9a0a1c-9c3f-4ee1-bd92-872c633a4687": "CO2 1st",
    "53695fca-36ef-4bfe-a937-f22c82829edc": "Temp 1st",
    "d89bf121-e1b6-409a-85d0-bfa9a9720000": "CO2 1st",
    "8b564222-b710-455f-95ba-f9d82ea67d60": "Temp 1st",
    "18030a11-6032-4439-8cf7-2e60f83264eb": "CO2 1st",
    "ec611817-5610-4ac1-9e72-f9d4c0a09e47": "Temp 1st",
    "235aea1e-be63-4d31-bec5-3005e15aa9cb": "CO2 1st",
    "ac96c34c-65ad-4d82-853a-4ad177be32c3": "Temp 1st",
    "eb73f46e-12a5-4509-a770-b38e0bf1885e": "CO2 1st",
    "b56c76f5-5ca2-42c3-a577-03fc534d4d32": "Temp 1st",
    "38f8d0b6-ca48-4db2-a403-36ffcb0caa02": "CO2 1st",
    "293109e1-54eb-4401-8672-e5ae0c9f76bd": "Temp 1st",
    "8acb30a3-e045-4968-b435-d392a3918f2f": "CO2 1st",
    "5d51aac9-7138-46ce-a73b-a6559e4ea93d": "Temp 1st",
    "19bac737-bcb2-4df0-9b7e-bb5753a431cd": "CO2 1st",
    "7aa8cf71-cf26-4a8e-a79b-88e1742adcf6": "Temp 1st",
    "a25e26bd-6777-4ebc-a773-ed11c036e179": "CO2 2nd",
    "fc3f3422-2af3-45a3-9b8b-4020218234cc": "Temp 2nd",
    "8b9b627a-3105-4503-b514-b2ea9776fc43": "CO2 2nd",
    "cd64e2f9-f93b-408f-af58-b68d93b44875": "Temp 2nd",
    "0bcc09c8-9069-4bdc-8947-0512a0143626": "CO2 2nd",
    "6f8d3525-0b9c-46b1-8797-dc1bcffab75f": "Temp 2nd",
    "3ee27086-c8a8-4e9b-bc60-8cdc2074e204": "CO2 2nd",
    "8f7b92a4-8866-47c8-a052-9739f284f149": "Temp 2nd",
    "a68853db-0dca-481e-99bc-47596fd3b360": "CO2 2nd",
    "52f5f39f-976a-42e1-b1ee-22bc13fb6eab": "Temp 2nd",
    "90494e02-1d47-411b-af0c-2485ab665fcc": "CO2 2nd",
    "74e702b6-4019-4619-9c62-d7ca8c3eb581": "Temp 2nd",
    "124ebe4d-b230-4384-850b-8b9636f6f0da": "CO2 2nd",
    "04882dbe-0168-41f6-8023-f2459f3beffc": "Temp 2nd",
    "ff900229-c6dd-435d-a356-85d2186679ca": "CO2 2nd",
    "3f9a8291-15f5-4eba-b138-4831a3a4d16c": "Temp 2nd",
    "9e62c9b1-8563-4854-8e97-3c10774822a9": "CO2 2nd",
    "3cf458e6-c5cf-4453-9a35-fa4f7a1c0b7a": "Temp 2nd"
}

df = pd.read_csv("data/TelemetryData.csv", names=["id", "timestamp", "value"], parse_dates=True)
df = df.replace(id_dictionary)  # Replace json ids with dictionary values


# Make individual tables
heat0 = df.loc[df["id"] == "Heat 0th floor (kWh)"]
heat1 = df.loc[df["id"] == "Heat 1st floor (kWh)"]
heat2 = df.loc[df["id"] == "Heat 2nd floor (kWh)"]

water0 = df.loc[df["id"] == "Water 0th floor (kWh)"]
water1 = df.loc[df["id"] == "Water 1st floor (kWh)"]
water2 = df.loc[df["id"] == "Water 2nd floor (kWh)"]

electricity = df.loc[df["id"] == "Electricity (kWh)"]
solar = df.loc[df["id"] == "Solar energy produced (Wh)"]
powerimport = df.loc[df["id"] == "Power imported from Grid (Wh)"]
powerexport = df.loc[df["id"] == "Power exported to Grid (Wh)"]

# Reset index to timestamp and rename columns
heat0 = heat0.rename(columns={"value": "Heat 0th floor (kWh)"}).set_index("timestamp")
del heat0["id"]

heat1 = heat1.rename(columns={"value": "Heat 1st floor (kWh)"}).set_index("timestamp")
del heat1["id"]

heat2 = heat2.rename(columns={"value": "Heat 2nd floor (kWh)"}).set_index("timestamp")
del heat2["id"]

water0 = water0.rename(columns={"value": "Water 0th floor (kWh)"}).set_index("timestamp")
del water0["id"]

water1 = water1.rename(columns={"value": "Water 1st floor (kWh)"}).set_index("timestamp")
del water1["id"]

water2 = water2.rename(columns={"value": "Water 2nd floor (kWh)"}).set_index("timestamp")
del water2["id"]

electricity = electricity.rename(columns={"value": "Electricity (kWh)"}).set_index("timestamp")
del electricity["id"]

solar = solar.rename(columns={"value": "Solar energy produced (Wh)"}).set_index("timestamp")
del solar["id"]

powerimport = powerimport.rename(columns={"value": "Power imported from Grid (Wh)"}).set_index("timestamp")
del powerimport["id"]

powerexport = powerexport.rename(columns={"value": "Power exported to Grid (Wh)"}).set_index("timestamp")
del powerexport["id"]




# Export to csv

def export_to_csv(table, filename):
    table.to_csv(filename)

export_to_csv(heat0, "heat-ground-floor.csv")
export_to_csv(heat1, "heat-1st-floor.csv")
export_to_csv(heat2, "heat-2nd-floor.csv")
export_to_csv(electricity, "electricity.csv")
export_to_csv(solar, "solar-produced.csv")
export_to_csv(powerimport, "power-imported-from-grid.csv")
export_to_csv(powerexport, "power-exported-to-grid.csv")

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
