#%%
import pandas as pd
df = pd.read_csv("data/TelemetryData.csv", names=["id", "timestamp", "value"], parse_dates=True)

