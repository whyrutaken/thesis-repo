import statistics as mf
from preprocessor import Preprocessor

df = Preprocessor(valid_to="2022-02-28").master_df



# %% Plots
mf.plot_solar_vs_consumption(df, "2020-06-21")
mf.plot_solar_vs_consumption(df, "2020-01-19")

mf.plot_autocorrelation(df, "2020-01-02", "2020-02-02")