import pandas as pd
import matplotlib.pyplot as plt
from Preprocessor import Preprocessor

# only reading the data, not changing it
class Printer:

    def __init__(self):
        self.df_hourly_res = Preprocessor("2022-02-28").df
        self.df_daily_res = self.get_daily_max_mean_min(
            ["solar_absolute", "imported_absolute", "exported_absolute", "consumption_absolute"])
        self.df_daily_max = pd.DataFrame([self.df_daily_res.max_solar_absolute, self.df_daily_res.max_imported_absolute, self.df_daily_res.max_exported_absolute, self.df_daily_res.max_consumption_absolute]).transpose()
        self.df_yearly_res = self.get_yearly_max_mean_min(
            ["solar_absolute", "imported_absolute", "exported_absolute", "consumption_absolute"])




    def get_daily_max_mean_min(self, list_of_columns: list) -> pd.DataFrame:
        dn = []
        for column in list_of_columns:
            max_col = "max_" + column
            mean_col = "mean_" + column
            min_col = "min_" + column

            max_ = self.df_hourly_res.resample('D')[column].max().to_frame()
            mean_ = self.df_hourly_res.resample('D')[column].mean().to_frame()
            min_ = self.df_hourly_res.resample('D')[column].min().to_frame()

            df = pd.concat([max_, mean_, min_], axis=1, ignore_index=True)
            df = df.rename(columns={0: max_col, 1: mean_col, 2: min_col})
            dn.append(df)
        dn = pd.concat(dn, axis=1)
        return dn

    def get_yearly_max_mean_min(self, list_of_columns: list) -> pd.DataFrame:
        dn = []
        for column in list_of_columns:
            df = self.df_hourly_res.loc[self.df_hourly_res.groupby(pd.Grouper(freq='D')).idxmax().loc[:,column]]
            max_col = "max_" + column
            mean_col = "mean_" + column
            min_col = "min_" + column

            max = df.resample('Y')[column].max().to_frame()
            mean = df.resample('Y')[column].mean().to_frame()
            min = df.resample('Y')[column].min().to_frame()

            df = pd.concat([max, mean, min], axis=1, ignore_index=True)
            df = df.rename(columns={0: max_col, 1: mean_col, 2: min_col})
            dn.append(df)
        dn = pd.concat(dn, axis=1)
        return dn

    def plot_solar_vs_consumption(self, date: str):
        self.df_hourly_res[date].solar_absolute.plot(legend=True)
        self.df_hourly_res[date].consumption_absolute.plot(legend=True)
        plt.show()

    def plot_autocorrelation(self, from_date: str, to_date: str):
        x = pd.plotting.autocorrelation_plot(self.df_hourly_res.loc[from_date:to_date].consumption_absolute)
        x.plot()
        plt.show()


printer = Printer()
