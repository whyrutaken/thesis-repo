import pandas as pd
import matplotlib.pyplot as plt


# only reading the data, not changing it
class Printer:

    def __init__(self):
        self.df_hourly_res = pd.read_csv("master_df.csv")
        self.df_daily_res = self.get_daily_max_mean_min(
            ["solar_absolute", "imported_absolute", "exported_absolute", "consumption_absolute"])
        self.df_daily_max = pd.DataFrame([self.df_daily_res.max_solar_absolute, self.df_daily_res.max_imported_absolute,
                                          self.df_daily_res.max_exported_absolute,
                                          self.df_daily_res.max_consumption_absolute]).transpose()
        self.df_yearly_res = self.get_yearly_max_mean_min(
            ["solar_absolute", "imported_absolute", "exported_absolute", "consumption_absolute"])

        self.df_seasonal_statistics = self.get_seasonal_max_mean_min()

        self.df_yearly_statistics = self.create_yearly_statistics_table()

    def create_yearly_statistics_table(self):
        dn = []
        for i in range(2):
            columns = ["max", "min", "mean", "std"]
            df = pd.DataFrame(self.df_yearly_res.iloc[i, 0:4]).transpose()
            df.columns = columns
            imported = pd.DataFrame(self.df_yearly_res.iloc[i, 4:8]).transpose()
            imported.columns = columns
            df = df.append(imported)
            exported = pd.DataFrame(self.df_yearly_res.iloc[i, 8:12]).transpose()
            exported.columns = columns
            df = df.append(exported)
            consumption = pd.DataFrame(self.df_yearly_res.iloc[i, 12:16]).transpose()
            consumption.columns = columns
            df = df.append(consumption)
            df.index = ["solar", "imported", "exported", "consumption"]
            dn.append(df)

        yearly_stat = pd.concat(dn, axis=1)
        yearly_stat.columns = ["max_2020", "min_2020", "mean_2020", "std_2020", "max_2021", "min_2021", "mean_2021",
                               "std_2021"]
        return yearly_stat

    def get_seasonal_max_mean_min(self):
        df = self.df_daily_max
        df['season'] = (df.index.month % 12 + 3) // 3

        seasons = {
            1: 'Winter',
            2: 'Spring',
            3: 'Summer',
            4: 'Autumn'
        }

        column_names = ["max_solar_absolute", "max_imported_absolute", "max_exported_absolute",
                        "max_consumption_absolute",
                        "min_solar_absolute", "min_imported_absolute", "min_exported_absolute",
                        "min_consumption_absolute",
                        "mean_solar_absolute", "mean_imported_absolute", "mean_exported_absolute",
                        "mean_consumption_absolute",
                        "std_solar_absolute", "std_imported_absolute", "std_exported_absolute",
                        "std_consumption_absolute"]

        index_2020 = ["Autumn_2020", "Spring_2020", "Summer_2020", "Winter_2020"]
        index_2021 = ["Autumn_2021", "Spring_2021", "Summer_2021", "Winter_2021"]
        df['season_name'] = df['season'].map(seasons)
        del df["season"]
        df_season_2020 = df[:"2020-12-31"].groupby(["season_name"]).max()
        df_season_2020 = pd.concat([df_season_2020, df[:"2020-12-31"].groupby(["season_name"]).mean()], axis=1)
        df_season_2020 = pd.concat([df_season_2020, df[:"2020-12-31"].groupby(["season_name"]).min()], axis=1)
        df_season_2020 = pd.concat([df_season_2020, df[:"2020-12-31"].groupby(["season_name"]).std()], axis=1)
        df_season_2020.columns = column_names
        df_season_2020.index = index_2020

        df_season_2021 = df["2021-01-01":"2021-12-31"].groupby(["season_name"]).max()
        df_season_2021 = pd.concat([df_season_2021, df["2021-01-01":"2021-12-31"].groupby(["season_name"]).mean()],
                                   axis=1)
        df_season_2021 = pd.concat([df_season_2021, df["2021-01-01":"2021-12-31"].groupby(["season_name"]).min()],
                                   axis=1)
        df_season_2021 = pd.concat([df_season_2021, df["2021-01-01":"2021-12-31"].groupby(["season_name"]).std()],
                                   axis=1)
        df_season_2021.columns = column_names
        df_season_2021.index = index_2021

        df_season = pd.concat([df_season_2020, df_season_2021], axis=0)
        df_season = df_season.transpose()
        df_season = df_season.loc[[
            "max_solar_absolute", "min_solar_absolute", "mean_solar_absolute", "std_solar_absolute",
            "max_imported_absolute", "min_imported_absolute", "mean_imported_absolute", "std_imported_absolute",
            "max_exported_absolute", "min_exported_absolute", "mean_exported_absolute", "std_exported_absolute",
            "max_consumption_absolute", "min_consumption_absolute", "mean_consumption_absolute", "std_consumption_absolute"],:]
        return df_season

    def get_daily_max_mean_min(self, list_of_columns: list) -> pd.DataFrame:
        dn = []
        for column in list_of_columns:
            max_col = "max_" + column
            min_col = "min_" + column
            mean_col = "mean_" + column
            std_col = "std_" + column

            max_ = self.df_hourly_res.resample('D')[column].max().to_frame()
            mean_ = self.df_hourly_res.resample('D')[column].mean().to_frame()
            min_ = self.df_hourly_res.resample('D')[column].min().to_frame()
            std_ = self.df_hourly_res.resample('D')[column].std().to_frame()

            df = pd.concat([max_, min_, mean_, std_], axis=1, ignore_index=True)
            df = df.rename(columns={0: max_col, 1: min_col, 2: mean_col, 3: std_col})
            dn.append(df)
        dn = pd.concat(dn, axis=1)
        return dn

    def get_yearly_max_mean_min(self, list_of_columns: list) -> pd.DataFrame:
        dn = []
        for column in list_of_columns:
            df = self.df_hourly_res.loc[self.df_hourly_res.groupby(pd.Grouper(freq='D')).idxmax().loc[:, column]]
            max_col = "max_" + column
            min_col = "min_" + column
            mean_col = "mean_" + column
            std_col = "std_" + column

            max_ = df.resample('Y')[column].max().to_frame()
            mean_ = df.resample('Y')[column].mean().to_frame()
            min_ = df.resample('Y')[column].min().to_frame()
            std_ = df.resample('Y')[column].std().to_frame()

            df = pd.concat([max_, min_, mean_, std_], axis=1, ignore_index=True)
            df = df.rename(columns={0: max_col, 1: min_col, 2: mean_col, 3: std_col})
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
