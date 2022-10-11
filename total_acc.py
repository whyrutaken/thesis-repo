#%%
import pandas as pd
import my_functions as mf


df_solar, df_imported, df_exported = mf.read_csv(["solar-produced.csv", "power-imported-from-grid.csv", "power-exported-to-grid.csv"], "2022-06-02")
df_solar["solar-daily-acc"] = mf.convert_total_to_daily_accumulation(df_solar)

