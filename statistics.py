import pandas as pd


# only reading the data, not changing it
class Statistics:

    def __init__(self):
        self.df_hourly_res = pd.read_csv("master_df.csv", parse_dates=True)

        self.df_daily_max = self.get_daily_max()
        self.df_yearly_total = self.get_yearly_total(
            ["solar_da", "imported_da", "exported_da", "consumption_da"])

        self.df_seasonal_total = self.get_seasonal_statistics("total", False,
                                                              ["max_solar_da", "max_imported_da", "max_exported_da",
                                                               "max_consumption_da",
                                                               "min_solar_da", "min_imported_da", "min_exported_da",
                                                               "min_consumption_da",
                                                               "mean_solar_da", "mean_imported_da",
                                                               "mean_exported_da", "mean_consumption_da",
                                                               "std_solar_da", "std_imported_da", "std_exported_da",
                                                               "std_consumption_da",
                                                               "total_solar_da", "total_imported_da",
                                                               "total_exported_da", "total_consumption_da"],
                                                              ["max_solar_da", "min_solar_da", "mean_solar_da",
                                                               "std_solar_da", "total_solar_da",
                                                               "max_imported_da", "min_imported_da",
                                                               "mean_imported_da", "std_imported_da",
                                                               "total_imported_da",
                                                               "max_exported_da", "min_exported_da",
                                                               "mean_exported_da", "std_exported_da",
                                                               "total_exported_da",
                                                               "max_consumption_da", "min_consumption_da",
                                                               "mean_consumption_da", "std_consumption_da",
                                                               "total_consumption_da"])
        self.df_seasonal_absolute = self.get_seasonal_statistics("absolute", False,
                                                                 ["max_solar_abs", "max_imported_abs",
                                                                  "max_exported_abs", "max_consumption_abs",
                                                                  "min_solar_abs", "min_imported_abs",
                                                                  "min_exported_abs", "min_consumption_abs",
                                                                  "mean_solar_abs", "mean_imported_abs",
                                                                  "mean_exported_abs", "mean_consumption_abs",
                                                                  "std_solar_abs", "std_imported_abs",
                                                                  "std_exported_abs", "std_consumption_abs"],
                                                                 ["max_solar_abs", "min_solar_abs", "mean_solar_abs",
                                                                  "std_solar_abs",
                                                                  "max_imported_abs", "min_imported_abs",
                                                                  "mean_imported_abs", "std_imported_abs",
                                                                  "max_exported_abs", "min_exported_abs",
                                                                  "mean_exported_abs", "std_exported_abs",
                                                                  "max_consumption_abs", "min_consumption_abs",
                                                                  "mean_consumption_abs", "std_consumption_abs"])

        self.df_seasonal_weekly_total = self.get_seasonal_statistics("total", True,
                                                                     ["wd_max_solar_da", "wd_max_imported_da",
                                                                      "wd_max_exported_da",
                                                                      "wd_max_consumption_da",
                                                                      "wd_min_solar_da", "wd_min_imported_da",
                                                                      "wd_min_exported_da",
                                                                      "wd_min_consumption_da",
                                                                      "wd_mean_solar_da", "wd_mean_imported_da",
                                                                      "wd_mean_exported_da", "wd_mean_consumption_da",
                                                                      "wd_std_solar_da", "wd_std_imported_da",
                                                                      "wd_std_exported_da",
                                                                      "wd_std_consumption_da",
                                                                      "we_max_solar_da", "we_max_imported_da",
                                                                      "we_max_exported_da",
                                                                      "we_max_consumption_da",
                                                                      "we_min_solar_da", "we_min_imported_da",
                                                                      "we_min_exported_da",
                                                                      "we_min_consumption_da",
                                                                      "we_mean_solar_da", "we_mean_imported_da",
                                                                      "we_mean_exported_da", "we_mean_consumption_da",
                                                                      "we_std_solar_da", "we_std_imported_da",
                                                                      "we_std_exported_da",
                                                                      "we_std_consumption_da",
                                                                      ],

                                                                     ["wd_max_solar_da", "we_max_solar_da",
                                                                      "wd_min_solar_da", "we_min_solar_da",
                                                                      "wd_mean_solar_da", "we_mean_solar_da",
                                                                      "wd_std_solar_da", "we_std_solar_da",
                                                                      "wd_max_imported_da", "we_max_imported_da",
                                                                      "wd_min_imported_da",
                                                                      "we_min_imported_da",
                                                                      "wd_mean_imported_da", "we_mean_imported_da",
                                                                      "wd_std_imported_da",
                                                                      "we_std_imported_da",
                                                                      "wd_max_exported_da", "we_max_exported_da",
                                                                      "wd_min_exported_da",
                                                                      "we_min_exported_da",
                                                                      "wd_mean_exported_da", "we_mean_exported_da",
                                                                      "wd_std_exported_da",
                                                                      "we_std_exported_da",
                                                                      "wd_max_consumption_da", "we_max_consumption_da",
                                                                      "wd_min_consumption_da",
                                                                      "we_min_consumption_da",
                                                                      "wd_mean_consumption_da",
                                                                      "we_mean_consumption_da",
                                                                      "wd_std_consumption_da",
                                                                      "we_std_consumption_da"])
        self.df_seasonal_weekly_absolute = self.get_seasonal_statistics("absolute", True,
                                                                        ["wd_max_solar_abs", "wd_max_imported_abs",
                                                                         "wd_max_exported_abs",
                                                                         "wd_max_consumption_abs",
                                                                         "wd_min_solar_abs", "wd_min_imported_abs",
                                                                         "wd_min_exported_abs",
                                                                         "wd_min_consumption_abs",
                                                                         "wd_mean_solar_abs", "wd_mean_imported_abs",
                                                                         "wd_mean_exported_abs",
                                                                         "wd_mean_consumption_abs",
                                                                         "wd_std_solar_abs", "wd_std_imported_abs",
                                                                         "wd_std_exported_abs",
                                                                         "wd_std_consumption_abs",
                                                                         "we_max_solar_abs", "we_max_imported_abs",
                                                                         "we_max_exported_abs",
                                                                         "we_max_consumption_abs",
                                                                         "we_min_solar_abs", "we_min_imported_abs",
                                                                         "we_min_exported_abs",
                                                                         "we_min_consumption_abs",
                                                                         "we_mean_solar_abs", "we_mean_imported_abs",
                                                                         "we_mean_exported_abs",
                                                                         "we_mean_consumption_abs",
                                                                         "we_std_solar_abs", "we_std_imported_abs",
                                                                         "we_std_exported_abs",
                                                                         "we_std_consumption_abs"],
                                                                        ["wd_max_solar_abs", "we_max_solar_abs",
                                                                         "wd_min_solar_abs",
                                                                         "we_min_solar_abs",
                                                                         "wd_mean_solar_abs", "we_mean_solar_abs",
                                                                         "wd_std_solar_abs",
                                                                         "we_std_solar_abs",
                                                                         "wd_max_imported_abs",
                                                                         "we_max_imported_abs",
                                                                         "wd_min_imported_abs", "we_min_imported_abs",
                                                                         "wd_mean_imported_abs",
                                                                         "we_mean_imported_abs",
                                                                         "wd_std_imported_abs", "we_std_imported_abs",

                                                                         "wd_max_exported_abs", "we_max_exported_abs",
                                                                         "wd_min_exported_abs",
                                                                         "we_min_exported_abs",
                                                                         "wd_mean_exported_abs", "we_mean_exported_abs",
                                                                         "wd_std_exported_abs",
                                                                         "we_std_exported_abs",
                                                                         "wd_max_consumption_abs",
                                                                         "we_max_consumption_abs",
                                                                         "wd_min_consumption_abs",
                                                                         "we_min_consumption_abs",
                                                                         "wd_mean_consumption_abs",
                                                                         "we_mean_consumption_abs",
                                                                         "wd_std_consumption_abs",
                                                                         "we_std_consumption_abs"])

        self.df_seasonal_metrics = self.get_metrics(self.df_seasonal_total, "total_solar_da", "total_exported_da",
                                                    "total_consumption_da")
        self.df_yearly_metrics = self.get_metrics(self.df_yearly_total, "solar_da", "exported_da",
                                                  "consumption_da")
        self.print_to_latex_in_kwh_rounded_up()

    def print_to_latex_in_kwh_rounded_up(self):
        pd.options.display.float_format = '{:,}'.format
        print(self.df_seasonal_total.div(1000).round(1).to_latex())
        print(self.df_seasonal_absolute.div(1000).round(1).to_latex())
        print(self.df_yearly_total.div(1000).round(1).to_latex())
        print(self.df_seasonal_metrics.mul(100).round(2).to_latex())
        print(self.df_seasonal_weekly_total.div(1000).round(1).to_latex())
        print(self.df_seasonal_weekly_absolute.div(1000).round(1).to_latex())

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

    @staticmethod
    def calculate_statistics(df: pd.DataFrame, begin_date: str, end_date: str, first_col: int, last_col: int) -> list:
        max_ = df[begin_date:end_date].groupby(["season_name"]).max().iloc[:, first_col:last_col]
        min_ = df[begin_date:end_date].groupby(["season_name"]).min().iloc[:, first_col:last_col]
        mean_ = df[begin_date:end_date].groupby(["season_name"]).mean().iloc[:, first_col:last_col]
        std_ = df[begin_date:end_date].groupby(["season_name"]).std().iloc[:, first_col:last_col]
        sum_ = df[begin_date:end_date].groupby(["season_name"]).sum().iloc[:, first_col:last_col]
        return [max_, min_, mean_, std_, sum_]

    def calculate_seasonal_everyday_statistics(self, type_: str, df: pd.DataFrame, begin_date: str, end_date: str,
                                               first_col: int, last_col: int, column_names,
                                               index_names) -> pd.DataFrame:
        statistics_ = self.calculate_statistics(df, begin_date, end_date, first_col, last_col)

        if type_ == "absolute":
            statistics_ = statistics_[:-1]

        df = pd.concat(statistics_, axis=1, ignore_index=True)
        df.columns = column_names
        df.index = index_names
        return df

    def calculate_seasonal_weekly_statistics(self, df: pd.DataFrame, begin_date: str, end_date: str,
                                             first_col: int, last_col: int, column_names, index_names) -> pd.DataFrame:
        df["day_of_week"] = df.index.dayofweek
        weekday = self.calculate_statistics(df.loc[(df["day_of_week"] < 5)], begin_date, end_date, first_col, last_col)[
                  :-1]
        weekend = self.calculate_statistics(df.loc[(df["day_of_week"] >= 5)], begin_date, end_date, first_col,
                                            last_col)[:-1]

        df_weekday = pd.concat(weekday, axis=1, ignore_index=True)
        df_weekend = pd.concat(weekend, axis=1, ignore_index=True)
        df = pd.concat([df_weekday, df_weekend], axis=1, ignore_index=True)
        df.columns = column_names
        df.index = index_names
        return df

    def get_seasonal_statistics(self, type_: str, weekly_division: bool, column_names: list,
                                ordered_column_names: list) -> pd.DataFrame:
        index_2020 = ["Autumn_2020", "Spring_2020", "Summer_2020", "Winter_2020"]
        index_2021 = ["Autumn_2021", "Spring_2021", "Summer_2021", "Winter_2021"]
        first_column = 0
        last_column = 0
        df = None
        if type_ == "total":
            df = self.get_season(self.df_daily_max)
            last_column = 4
        if type_ == "absolute":
            first_column = 4
            last_column = 8
            df = self.get_season(self.df_hourly_res)

        if weekly_division:
            df_season_2020 = self.calculate_seasonal_weekly_statistics(df, "2020-01-01", "2020-12-31", first_column,
                                                                       last_column, column_names, index_2020)

            df_season_2021 = self.calculate_seasonal_weekly_statistics(df, "2021-01-01", "2021-12-31", first_column,
                                                                       last_column, column_names, index_2021)
        else:
            df_season_2020 = self.calculate_seasonal_everyday_statistics(type_, df, "2020-01-01", "2020-12-31",
                                                                         first_column, last_column,
                                                                         column_names, index_2020)
            df_season_2021 = self.calculate_seasonal_everyday_statistics(type_, df, "2021-01-01", "2021-12-31",
                                                                         first_column, last_column,
                                                                         column_names, index_2021)
        df = pd.concat([df_season_2020, df_season_2021], axis=0)
        df = df.transpose()
        df = df.loc[ordered_column_names, :]
        return df

    def get_daily_max(self) -> pd.DataFrame:
        return self.df_hourly_res.resample('D').max()

    def get_yearly_total(self, list_of_columns: list) -> pd.DataFrame:
        temp = []
        for column in list_of_columns:
            temp.append(self.df_daily_max.resample('Y')[column].sum().to_frame())
        return pd.concat(temp, axis=1).transpose()

    @staticmethod
    def calculate_max_self_sufficiency(df: pd.DataFrame, solar_index: str, consumption_index: str) -> pd.Series:
        return df.loc[solar_index] / df.loc[consumption_index]

    @staticmethod
    def calculate_self_sufficiency(df: pd.DataFrame, solar_index: str, exported_index: str,
                                   consumption_index: str) -> pd.Series:
        return (df.loc[solar_index] - df.loc[exported_index]) / df.loc[consumption_index]

    @staticmethod
    def calculate_self_consumption(df: pd.DataFrame, solar_index: str, exported_index: str) -> pd.Series:
        return (df.loc[solar_index] - df.loc[exported_index]) / \
               df.loc[solar_index]

    @staticmethod
    def calculate_sold_solar(df: pd.DataFrame, solar_index: str, exported_index: str) -> pd.Series:
        return df.loc[exported_index] / df.loc[solar_index]

    def get_metrics(self, df: pd.DataFrame, solar_index: str, exported_index: str,
                    consumption_index: str) -> pd.DataFrame:
        max_self_sufficiency = self.calculate_max_self_sufficiency(df, solar_index, consumption_index)
        self_sufficiency = self.calculate_self_sufficiency(df, solar_index, exported_index, consumption_index)
        self_consumption = self.calculate_self_consumption(df, solar_index, exported_index)
        sold_solar = self.calculate_sold_solar(df, solar_index, exported_index)

        df = pd.concat([max_self_sufficiency, self_sufficiency, self_consumption, sold_solar], axis=1,
                       ignore_index=True)
        df.columns = ["max_self_sufficiency", "self_sufficiency", "self_consumption", "sold_solar_percentage"]
        return df.transpose()


statistics = Statistics()
