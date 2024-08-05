import numpy as np
import pandas as pd
import os

# Flags for results
ORIGINAL = "original"
BEST = "best_adj"
RULE = "rule_based_adj"
NAIVE_MPC = "naive_mpc_adj"
STOCHASTIC_MPC = "stochastic_mpc_adj"

# Constants
HOURS_PER_DAY = 24  # 24 hours
HOURS_PER_WEEK = 168  # 7 days
HOURS_PER_MONTH = 720  # 30 days
HOURS_PER_YEAR = 8640  # 12 months (360 days)

M = 1000 # A large number

NOMINAL_WIND = 10  # Nominal wind power

RHO_H = 20  # Power to hydrogen efficiency of the electrolyzer [kg/MWh]
ETA_STORAGE = 0.88  # Storage efficiency of the hydrogen storage [kg/MWh]
LAMBDA_H = 2.5  # Hydrogen price [€/kg]
H_MIN = 50  # Minimum daily hydrogen production [kg]
P_H_MAX = 10  # Maximal power consumption capacity of the electrolyzer [MWh]
PRICE_H = 35.2  # Hydrogen price [€/MWh]

PENALTY = 80.61  # Value equal to the 95% quantile of the buy prices

# Top domain value (linked to price domains)
TOP_DOMAIN = 53.32  # 90%

# Main datafile
DATAFILE = "../data/2022_2023/2022_2023_data.csv"

ROLLING_FORECASTS_FILE = "../forecasting/rolling_forecasts_2019_2020.csv"

RESULTS_DIR = os.path.join(os.getcwd(), "../results/models")

Q_FORECAST_CALCULATED = [12.195654545757634, 0.5299454470522954, 1.2367673427003123, -0.5444726493505923,
                         4.9381332869069965]
Q_INTERCEPT_CALCULATED = -0.13315441932264693


# def import_consts(start_date="2020-01-01", end_date="2022-01-01", negative_prices=False):
#     """
#     Import data and set constants for the analysis.
#
#     Args:
#         start_date (str): Start date of the analysis.
#         end_date (str): End date of the analysis.
#         negative_prices (bool): If True, allows for negative prices.
#
#     Returns:
#         tuple: Various datasets and constants used in the analysis.
#     """
#     all_data = pd.read_csv(DATAFILE)
#     all_data.index = pd.date_range(start="2020-01-01", periods=len(all_data), freq="h")
#     all_data = all_data.loc[start_date:end_date]
#     prices_B = np.maximum(all_data["UP"].to_numpy(), 0)
#     prices_S = np.maximum(all_data["DW"].to_numpy(), 0)
#     prices_F = np.maximum(all_data["forward_RE"].to_numpy(), 0)
#     prices_forecast = np.maximum(all_data["forward_FC"].to_numpy(), 0)
#
#     features = all_data.loc[:, ["Offshore DK2", "Offshore DK1", "Onshore DK2", "Onshore DK1", "production_FC"]]
#     features["forward"] = prices_F
#     features_red = all_data.loc[:, ["production_FC"]]
#     features_red["forward"] = prices_F
#     realized = all_data.loc[:, "production_RE"].to_numpy()
#     realized *= NOMINAL_WIND
#
#     fm_df = pd.read_csv("./results/2020_forecast_model.csv")
#     forecast_production = fm_df.loc[:, "forecast_production"]
#
#     return (
#         prices_B, prices_S, prices_F, prices_forecast, features,
#         features_red, realized, PRICE_H, P_H_MAX, NOMINAL_WIND,
#         PENALTY, H_MIN, forecast_production
#     )