#%%

import pandas as pd
import statistics as mf

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



#%%
# TODO: remove this function
def create_yearly_statistics_table(self):
    dn = []
    for i in range(2):
        columns = ["max", "min", "mean", "std", "total"]
        df = pd.DataFrame(self.df_yearly_res.iloc[i, 0:5]).transpose()
        df.columns = columns
        imported = pd.DataFrame(self.df_yearly_res.iloc[i, 5:10]).transpose()
        imported.columns = columns
        df = df.append(imported)
        exported = pd.DataFrame(self.df_yearly_res.iloc[i, 10:15]).transpose()
        exported.columns = columns
        df = df.append(exported)
        consumption = pd.DataFrame(self.df_yearly_res.iloc[i, 15:20]).transpose()
        consumption.columns = columns
        df = df.append(consumption)

        df.index = ["solar", "imported", "exported", "consumption"]
        dn.append(df)

    yearly_stat = pd.concat(dn, axis=1)
    yearly_stat.columns = ["max_2020", "min_2020", "mean_2020", "std_2020", "total_2020", "max_2021", "min_2021",
                           "mean_2021",
                           "std_2021", "total_2021"]
    return yearly_stat