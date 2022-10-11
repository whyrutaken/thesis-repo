# %%
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

    def make_tables(self, df: pd.DataFrame, id_list: list) -> list:
        list_of_dfs = list()
        for id_ in id_list:
            table = df.loc[df["id"] == id_]
            table = table.rename(columns={"value": id_}).set_index("timestamp")
            table.index = pd.to_datetime(table.index)  # set index to DateTimeIndex
            table = table.reindex(index=table.index[::-1])  # reverse df
            del table["id"]
            list_of_dfs.append(table)

        return list_of_dfs

    def get_new_resolution_by_argument(self, df: pd.DataFrame, resolution: str) -> pd.DataFrame:
        resampled_df = df.resample(resolution).max()
        return resampled_df.fillna(value=0)

    def set_df_valid_date(self, df: pd.DataFrame, date: str) -> pd.DataFrame:
        return df[:date]  # eliminate rows after 2022-03-01

    def get_abs_value_from_daily_acc(self, df: pd.DataFrame, old_column: str, new_column: str) -> pd.DataFrame:
        df[new_column] = df[old_column].diff().fillna(0)  # get the difference of elements from previous elements

        # get date change locations where diff != 0 Days
        # (in df2["add"], True: whenever the day changed, False: whenever is a value from the same day)
        df["date"] = df.index.date
        df["add"] = df["date"].diff().ne("0D")
        # 3. add the previous total back
        df.loc[df["add"], new_column] += df[old_column].shift()[df["add"]]
        df = df.fillna(0)
        del df["date"]
        del df["add"]
        return df

    def calculate_consumption(self, imported: float, exported: float, solar: float) -> float:
        used = imported + solar
        return used - exported

    # TODO: instead of deleting completely, you could inject the value from an hour before
    def del_lines(self, df: pd.DataFrame, list_of_dates: list) -> pd.DataFrame:
        for date in list_of_dates:
            df = df[~(df.index == date)]
        return df

    def create_master_df(self):
        SDA = "solar_daily_acc"
        SA = "solar_absolute"
        IDA = "imported_daily_acc"
        EDA = "exported_daily_acc"
        CDA = "consumption_daily_acc"
        CA = "consumption_absolute"

        df = self.df_solar_resampled.copy()
        df = df.rename(columns={"Solar energy produced (Wh)": SDA})
        df[IDA] = self.df_power_imported_resampled.iloc[:, 0]
        df[EDA] = self.df_power_exported_resampled.iloc[:, 0]
        df[CDA] = df.apply(
            lambda row: self.calculate_consumption(row[IDA], row[EDA], row[SDA]), axis=1)

        df = self.get_abs_value_from_daily_acc(df, SDA, SA)
        df = self.get_abs_value_from_daily_acc(df, CDA, CA)

        df = self.del_lines(df, ["2020-01-01 00:00", "2020-03-29 02:00", "2021-03-28 02:00"])
        return df

    def __init__(self, valid_to: str):
        # telemetry data to separate tables
        df_telemetry = pd.read_csv("data/TelemetryData.csv", names=["id", "timestamp", "value"], parse_dates=True)
        df_telemetry = df_telemetry.replace(self.json_id_dictionary)  # Replace json ids with dictionary values

        self.df_solar, self.df_power_imported, self.df_power_exported = self.make_tables(
            df_telemetry, [self.solar_id, self.power_imported_id, self.power_exported_id])

        # hourly resolution/resample
        self.df_solar_resampled = self.get_new_resolution_by_argument(self.df_solar, "H")
        self.df_power_imported_resampled = self.get_new_resolution_by_argument(self.df_power_imported, 'H')
        self.df_power_exported_resampled = self.get_new_resolution_by_argument(self.df_power_exported, 'H')

        # valid until date "2022-02-28"
        self.set_df_valid_date(self.df_solar_resampled, valid_to)
        self.set_df_valid_date(self.df_power_imported_resampled, valid_to)
        self.set_df_valid_date(self.df_power_exported_resampled, valid_to)

        self.df = self.create_master_df()

    def export_raw_data(self):
        self.df_solar.to_csv("solar-produced.csv")
        self.df_power_imported.to_csv("power-imported-from-grid.csv")
        self.df_power_exported.to_csv("power-exported-to-grid.csv")


preprocessor = Preprocessor("2022-02-28")
