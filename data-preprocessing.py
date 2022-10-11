import my_functions as mf

df_solar, df_power_imported, df_power_exported = mf.read_csv(
    ["solar-produced.csv", "power-imported-from-grid.csv", "power-exported-to-grid.csv"], "2022-02-28")

df_resampled_solar = mf.get_new_resolution_by_argument(df_solar, "H")
df_resampled_power_imported = mf.get_new_resolution_by_argument(df_power_imported, "H")
df_resampled_power_exported = mf.get_new_resolution_by_argument(df_power_exported, "H")


SDA = "solar_daily_acc"
SA = "solar_absolute"
IDA = "imported_daily_acc"
EDA = "exported_daily_acc"
CDA = "consumption_daily_acc"
CA = "consumption_absolute"

df = df_resampled_solar.copy()
df = df.rename(columns={"Solar energy produced (Wh)": SDA})
df[IDA] = df_resampled_power_imported.iloc[:, 0]
df[EDA] = df_resampled_power_exported.iloc[:, 0]
df[CDA] = df.apply(
    lambda row: mf.calculate_consumption(row[IDA], row[EDA], row[SDA]), axis=1)

df = mf.get_abs_value_from_daily_acc(df, SDA, SA)
df = mf.get_abs_value_from_daily_acc(df, CDA, CA)


df = mf.del_lines(df, ["2020-01-01 00:00", "2020-03-29 02:00", "2021-03-28 02:00"])


# %% Plots
mf.plot_solar_vs_consumption(df, "2020-06-21")
mf.plot_solar_vs_consumption(df, "2020-01-19")

mf.plot_autocorrelation(df, "2020-01-02", "2020-02-02")