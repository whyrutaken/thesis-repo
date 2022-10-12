#%%

import pandas as pd
import Printer as mf

# deprecated
def convert_total_to_daily_accumulation(df: pd.DataFrame) -> pd.DataFrame:
    # BUT modify read_csv() to make it work
    # for rows after 2022-03-01

    new_df = df["2022-03-01":"2022-06-02"]
    new_df = new_df.iloc[::-1]
    new_df = new_df.diff(periods=1).fillna(new_df)
    return new_df

df_solar, df_imported, df_exported = mf.read_csv(["solar-produced.csv", "power-imported-from-grid.csv", "power-exported-to-grid.csv"], "2022-06-02")
df_solar_daily_acc = mf.convert_total_to_daily_accumulation(df_solar).iloc[:,0]

