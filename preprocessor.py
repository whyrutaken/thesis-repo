#####
#   Class for preprocessing the data
#

import pandas as pd


class Preprocessor:
    json_id_dictionary = {
        # "af8313d0-22c2-49bc-a736-8cde9d6b6415": "Heat 0th floor (kWh)",
        # "e28aa982-3ab8-4be2-9954-8c0ee699dceb": "Water 0th floor (kWh)",
        # "5c10d36c-99e6-49b5-af06-0df5b9985b71": "Heat 1st floor (kWh)",
        # "5419cd17-8f71-4139-afa1-dec0bbfd21a4": "Water 1st floor (kWh)",
        # "481b48aa-5bba-44ae-a5ab-2c78c68bfcc5": "Heat 2nd floor (kWh)",
        # "00a46798-af26-49cd-84e9-fd64a69cbe24": "Water 2nd floor (kWh)",
        # "70ccaf33-1575-4bd5-ae4b-3cb6d857731a": "Electricity (kWh)",
        "bbc8c07b-b8e0-47e7-a4f3-7fb8d4768260": "Solar energy produced (Wh)",
        "b007fc66-9715-4e5a-b4dc-1540c92de99e": "Power imported from Grid (Wh)",
        "40299d14-bf30-4747-8f72-edfe7c26c15a": "Power exported to Grid (Wh)"
    }

    solar_id = "Solar energy produced (Wh)"
    power_imported_id = "Power imported from Grid (Wh)"
    power_exported_id = "Power exported to Grid (Wh)"

    def __init__(self, valid_to="2022-02-28"):
        pd.set_option('display.precision', 1)

        self.df_solar, self.df_power_imported, self.df_power_exported = self.create_tables_and_set_index(
            [self.solar_id, self.power_imported_id, self.power_exported_id])
        self.df_solar_resampled, self.df_power_imported_resampled, self.df_power_exported_resampled = \
            self.get_hourly_resolution([self.df_solar, self.df_power_imported, self.df_power_exported])
        self.master_df = self.create_master_df(valid_to)

        self.export([self.master_df, self.df_solar, self.df_power_imported, self.df_power_exported],
                    ["extracted-data/master-df.csv", "extracted-data/solar-produced.csv",
                     "extracted-data/power-imported.csv", "extracted-data/power-exported.csv"])

    def read_telemetry_data(self) -> pd.DataFrame:
        df_telemetry = pd.read_csv("initial-data/TelemetryData.csv", names=["id", "timestamp", "value"],
                                   parse_dates=True)
        return df_telemetry.replace(self.json_id_dictionary)

    def create_tables_and_set_index(self, id_list: list) -> list:
        df = self.read_telemetry_data()
        df_list = list()
        for id_ in id_list:
            table = df.loc[df["id"] == id_]
            table = table.rename(columns={"value": id_})
            table = table.set_index("timestamp")
            table.index = pd.to_datetime(table.index)  # set index to DateTimeIndex
            table = table.reindex(index=table.index[::-1])  # reverse df
            del table["id"]
            df_list.append(table)
        return df_list

    @staticmethod
    def get_abs_value_from_daily_acc(df: pd.DataFrame, old_column: str, new_column: str) \
            -> pd.DataFrame:
        df[new_column] = df[old_column].diff().fillna(0)  # get the difference of elements from previous elements

        # get date change locations where diff != 0 Days
        # (in df2["add"], True: whenever the day changed, False: whenever is a value from the same day)
        df["date"] = df.index.date
        df["add"] = df["date"].diff().ne("0D")
        # 3. add the previous total back
        df.loc[df["add"], new_column] += df[old_column].shift()[df["add"]]
        del df["date"]
        del df["add"]
        return df.fillna(0)

    @staticmethod
    def get_new_resolution_by_argument(df: pd.DataFrame, resolution: str) \
            -> pd.DataFrame:
        return df.resample(resolution).max().fillna(value=0)

    def get_hourly_resolution(self, df_list: list) -> list:
        return_list = []
        for df in df_list:
            return_list.append(self.get_new_resolution_by_argument(df, 'H'))
        return return_list

    def create_master_df(self, valid_to: str) -> pd.DataFrame:
        SDA = "solar_da"
        IDA = "imported_da"
        EDA = "exported_da"
        DDA = "demand_da"

        SA = "solar_absolute"
        IA = "imported_absolute"
        EA = "exported_absolute"
        DA = "demand_absolute"

        master_df = self.df_solar_resampled.copy()
        master_df = master_df.rename(columns={"Solar energy produced (Wh)": SDA})
        master_df[IDA] = self.df_power_imported_resampled.iloc[:, 0]
        master_df[EDA] = self.df_power_exported_resampled.iloc[:, 0]
        master_df[DDA] = master_df.apply(
            lambda row: self.calculate_demand(row[IDA], row[EDA], row[SDA]), axis=1)

        master_df = self.get_abs_value_from_daily_acc(master_df, SDA, SA)
        master_df = self.get_abs_value_from_daily_acc(master_df, IDA, IA)
        master_df = self.get_abs_value_from_daily_acc(master_df, EDA, EA)
        master_df = self.get_abs_value_from_daily_acc(master_df, DDA, DA)

     #   master_df = self.del_lines(master_df, ["2020-01-01 00:00", "2020-03-29 02:00", "2021-03-28 02:00"])
        master_df = self.set_df_valid_date(master_df, valid_to)
        master_df.index = pd.to_datetime(master_df.index)
        master_df = self.fill_missing_values(master_df, ["2020-03-29 02:00", "2021-03-28 02:00"])
        return master_df

    @staticmethod
    def calculate_demand(imported: float, exported: float, solar: float) -> float:
        return imported + solar - exported

    # TODO: instead of deleting completely, you could inject the value from an hour before
    @staticmethod
    def del_lines(df: pd.DataFrame, list_of_dates: list) -> pd.DataFrame:
        for date in list_of_dates:
            df = df[~(df.index == date)]
        return df


    # copy the value from the same time the day before (for absolute values)
    # zero for daily aggregated values
    def fill_missing_values(self, df: pd.DataFrame, timestamps: list) -> pd.DataFrame:
        for timestamp in timestamps:
            observation_index = df.index.get_loc(timestamp)
            df.loc[timestamp, "solar_da":"demand_da"] = 0
            df.loc[timestamp, "solar_absolute":"demand_absolute"] = df.iloc[(observation_index-24)]
        return df

    @staticmethod
    def set_df_valid_date(df: pd.DataFrame, date: str) -> pd.DataFrame:
        return df[:date]  # eliminate rows after 2022-03-01

    @staticmethod
    def export(df_list: list, file_list: list):
        for df, filename in zip(df_list, file_list):
            df.to_csv(filename, index_label=False)


preprocessor = Preprocessor("2022-02-28")
