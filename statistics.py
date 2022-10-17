import pandas as pd


# only reading the data, not changing it
class Statistics:

    def __init__(self):
        self.df_hourly_res = pd.read_csv("master_df.csv", parse_dates=True)

        self.df_daily_max = self.get_daily_max()
        self.df_yearly_total = self.get_yearly_total(
            ["solar_da", "imported_da", "exported_da", "consumption_da"])

        self.df_seasonal_total_acc = self.get_seasonal_statistics("total_acc",
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
        self.df_seasonal_absolute = self.get_seasonal_statistics("absolute",
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

        self.print_to_latex_in_kwh_rounded_up()
        self.yearly_self_consumption = self.calculate_self_consumption(self.df_yearly_total, "solar_da",
                                                                       "exported_da")
        self.yearly_self_sufficiency = self.calculate_self_sufficiency(self.df_yearly_total, "solar_da",
                                                                       "exported_da", "consumption_da")
        self.seasonal_self_consumption = self.calculate_self_consumption(self.df_seasonal_total_acc, "total_solar_da",
                                                                         "total_exported_da")

    def print_to_latex_in_kwh_rounded_up(self):
        pd.options.display.float_format = '{:,}'.format
        print(self.df_seasonal_total_acc.div(1000).round(1).to_latex())
        print(self.df_seasonal_absolute.div(1000).round(1).to_latex())
        print(self.df_yearly_total.div(1000).round(1).to_latex())

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
    def calculate_seasonal_statistics(df: pd.DataFrame, begin_date: str, end_date: str, first_col: int, last_col: int,
                                      column_names, index_names) -> pd.DataFrame:
        max_ = df[begin_date:end_date].groupby(["season_name"]).max().iloc[:, first_col:last_col]
        min_ = df[begin_date:end_date].groupby(["season_name"]).min().iloc[:, first_col:last_col]
        mean_ = df[begin_date:end_date].groupby(["season_name"]).mean().iloc[:, first_col:last_col]
        std_ = df[begin_date:end_date].groupby(["season_name"]).std().iloc[:, first_col:last_col]
        sum_ = df[begin_date:end_date].groupby(["season_name"]).sum().iloc[:, first_col:last_col]
        if len(column_names) == 20:
            df = pd.concat([max_, min_, mean_, std_, sum_], axis=1, ignore_index=True)
            df.columns = column_names
            df.index = index_names
            return df
        else:
            df = pd.concat([max_, min_, mean_, std_], axis=1, ignore_index=True)
            df.columns = column_names
            df.index = index_names
            return df

    def get_seasonal_statistics(self, type_: str, column_names: list, ordered_column_names: list) -> pd.DataFrame:
        index_2020 = ["Autumn_2020", "Spring_2020", "Summer_2020", "Winter_2020"]
        index_2021 = ["Autumn_2021", "Spring_2021", "Summer_2021", "Winter_2021"]
        first_column = 0
        last_column = 0
        df = None
        if type_ == "total_acc":
            df = self.get_season(self.df_daily_max)
            last_column = 4
        if type_ == "absolute":
            first_column = 4
            last_column = 8
            df = self.get_season(self.df_hourly_res)

        df_season_2020 = self.calculate_seasonal_statistics(df, "2020-01-01", "2020-12-31", first_column, last_column,
                                                            column_names, index_2020)
        df_season_2021 = self.calculate_seasonal_statistics(df, "2021-01-01", "2021-12-31", first_column, last_column,
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
    def calculate_max_self_sufficiency(df: pd.DataFrame, solar_index: str, consumption_index: str):
        return df.loc[solar_index] / df.loc[consumption_index]

    @staticmethod
    def calculate_self_sufficiency(df: pd.DataFrame, solar_index: str, exported_index: str, consumption_index: str):
        return (df.loc[solar_index] - df.loc[exported_index]) / df.loc[consumption_index]

    @staticmethod
    def calculate_self_consumption(df: pd.DataFrame, solar_index: str, exported_index: str):
        return (df.loc[solar_index] - df.loc[exported_index]) / \
               df.loc[solar_index]

    @staticmethod
    def calculate_sold_solar(df: pd.DataFrame, solar_index: str, exported_index: str):
        return df.loc[exported_index] / df.loc[solar_index]


statistics = Statistics()
