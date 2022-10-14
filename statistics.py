import pandas as pd
import matplotlib.pyplot as plt


# only reading the data, not changing it
class Statistics:

    def __init__(self):
        self.df_hourly_res = pd.read_csv("master_df.csv", parse_dates=True)

        self.df_daily_max = self.get_daily_max()
        self.df_yearly_total = self.get_yearly_total(
            ["solar_daily_acc", "imported_daily_acc", "exported_daily_acc", "consumption_daily_acc"])

        self.df_seasonal_total_acc = self.get_seasonal_total_acc_statistics()
        self.df_seasonal_absolute = self.get_seasonal_absolute_statistics()

        self.print_to_latex()

    def print_to_latex(self):
        pd.options.display.float_format = '{:20,.0f}'.format
        print(self.df_seasonal_total_acc.to_latex())
        print(self.df_seasonal_absolute.to_latex())
        print(self.df_yearly_total.to_latex())

    @staticmethod
    def get_season(df: pd.DataFrame) -> pd.DataFrame:
        df['season'] = (df.index.month % 12 + 3) // 3

        seasons = {
            1: 'Winter',
            2: 'Spring',
            3: 'Summer',
            4: 'Autumn'
        }
        df['season_name'] = df['season'].map(seasons)
        del df["season"]
        return df

    def get_seasonal_absolute_statistics(self):
        df = self.get_season(self.df_hourly_res)
        column_names = ["max_solar_absolute", "max_imported_absolute", "max_exported_absolute", "max_consumption",
                        "min_solar_absolute", "min_imported_absolute", "min_exported_absolute",
                        "min_consumption",
                        "mean_solar_absolute", "mean_imported_absolute", "mean_exported_absolute",
                        "mean_consumption",
                        "std_solar_absolute", "std_imported_absolute", "std_exported_absolute",
                        "std_consumption"]
        index_2020 = ["Autumn_2020", "Spring_2020", "Summer_2020", "Winter_2020"]
        index_2021 = ["Autumn_2021", "Spring_2021", "Summer_2021", "Winter_2021"]

        max_ = df[:"2020-12-31"].groupby(["season_name"]).max().iloc[:, 4:]
        min_ = df[:"2020-12-31"].groupby(["season_name"]).min().iloc[:, 4:]
        mean_ = df[:"2020-12-31"].groupby(["season_name"]).mean().iloc[:, 4:]
        std_ = df[:"2020-12-31"].groupby(["season_name"]).std().iloc[:, 4:]
        df_season_2020 = pd.concat([max_, min_, mean_, std_], axis=1, ignore_index=True)
        df_season_2020.columns = column_names
        df_season_2020.index = index_2020

        max_ = df["2021-01-01":"2021-12-31"].groupby(["season_name"]).max().iloc[:, 4:]
        min_ = df["2021-01-01":"2021-12-31"].groupby(["season_name"]).min().iloc[:, 4:]
        mean_ = df["2021-01-01":"2021-12-31"].groupby(["season_name"]).mean().iloc[:, 4:]
        std_ = df["2021-01-01":"2021-12-31"].groupby(["season_name"]).std().iloc[:, 4:]
        df_season_2021 = pd.concat([max_, min_, mean_, std_], axis=1, ignore_index=True)
        df_season_2021.columns = column_names
        df_season_2021.index = index_2021

        df_season = pd.concat([df_season_2020, df_season_2021], axis=0)
        df = df_season.transpose()
        df = df.loc[[
                        "max_solar_absolute", "min_solar_absolute", "mean_solar_absolute",
                        "std_solar_absolute",
                        "max_imported_absolute", "min_imported_absolute", "mean_imported_absolute",
                        "std_imported_absolute",
                        "max_exported_absolute", "min_exported_absolute", "mean_exported_absolute",
                        "std_exported_absolute",
                        "max_consumption", "min_consumption",
                        "mean_consumption", "std_consumption",
                    ], :]
        return df

    def get_seasonal_total_acc_statistics(self):
        df = self.get_season(self.df_daily_max)

        column_names = ["max_solar_da", "max_imported_da", "max_exported_da", "max_consumption_da",
                        "min_solar_da", "min_imported_da", "min_exported_da",
                        "min_consumption_da",
                        "mean_solar_da", "mean_imported_da", "mean_exported_da",
                        "mean_consumption_da",
                        "std_solar_da", "std_imported_da", "std_exported_da",
                        "std_consumption_da",
                        "total_solar_da", "total_imported_da", "total_exported_da",
                        "total_consumption_da"]

        index_2020 = ["Autumn_2020", "Spring_2020", "Summer_2020", "Winter_2020"]
        index_2021 = ["Autumn_2021", "Spring_2021", "Summer_2021", "Winter_2021"]

        max_ = df[:"2020-12-31"].groupby(["season_name"]).max().iloc[:, :4]
        min_ = df[:"2020-12-31"].groupby(["season_name"]).min().iloc[:, :4]
        mean_ = df[:"2020-12-31"].groupby(["season_name"]).mean().iloc[:, :4]
        std_ = df[:"2020-12-31"].groupby(["season_name"]).std().iloc[:, :4]
        sum_ = df[:"2020-12-31"].groupby(["season_name"]).sum().iloc[:, :4]
        df_season_2020 = pd.concat([max_, min_, mean_, std_, sum_], axis=1, ignore_index=True)
        df_season_2020.columns = column_names
        df_season_2020.index = index_2020

        max_ = df["2021-01-01":"2021-12-31"].groupby(["season_name"]).max().iloc[:, :4]
        min_ = df["2021-01-01":"2021-12-31"].groupby(["season_name"]).min().iloc[:, :4]
        mean_ = df["2021-01-01":"2021-12-31"].groupby(["season_name"]).mean().iloc[:, :4]
        std_ = df["2021-01-01":"2021-12-31"].groupby(["season_name"]).std().iloc[:, :4]
        sum_ = df["2021-01-01":"2021-12-31"].groupby(["season_name"]).sum().iloc[:, :4]
        df_season_2021 = pd.concat([max_, min_, mean_, std_, sum_], axis=1, ignore_index=True)
        df_season_2021.columns = column_names
        df_season_2021.index = index_2021

        df_season = pd.concat([df_season_2020, df_season_2021], axis=0)
        df = df_season.transpose()
        df = df.loc[[
                        "max_solar_da", "min_solar_da", "mean_solar_da",
                        "std_solar_da", "total_solar_da",
                        "max_imported_da", "min_imported_da", "mean_imported_da",
                        "std_imported_da", "total_imported_da",
                        "max_exported_da", "min_exported_da", "mean_exported_da",
                        "std_exported_da", "total_exported_da",
                        "max_consumption_da", "min_consumption_da",
                        "mean_consumption_da", "std_consumption_da",
                        "total_consumption_da"], :]
        return df

    def get_daily_max(self) -> pd.DataFrame:
        return self.df_hourly_res.resample('D').max()

    def get_yearly_total(self, list_of_columns: list) -> pd.DataFrame:
        dn = []
        for column in list_of_columns:
            df = self.df_daily_max
            sum_ = df.resample('Y')[column].sum().to_frame()
            dn.append(sum_)
        dn = pd.concat(dn, axis=1).transpose()
        return dn


statistics = Statistics()
