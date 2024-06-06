import numpy as np
import pandas as pd

# Constants
HOURS_PER_DAY = 24 # 24 hours
HOURS_PER_WEEK = 168  # 7 days
HOURS_PER_MONTH = 720  # 30 days
HOURS_PER_YEAR = 8760  # 365 days

MAX_WIND = 10  # Maximum wind power
NOMINAL_WIND = 10  # Nominal wind power

rho_H = 20  # Power to hydrogen efficiency of the electrolyzer [kg/MWh]
eta_storage = 0.88  # Storage efficiency of the hydrogen storage [kg/MWh]
lambda_H = 2.5  # Hydrogen price [€/kg]
H_MIN = 50  # Minimum daily hydrogen production [kg]
P_H = 10  # Maximal power consumption capacity of the electrolyzer [MWh]
PRICE_H = 35.2  # Hydrogen price [€/MWh]

# Define color constants using RGB normalized to [0, 1] scale
RED = (0.77, 0, 0.05)  # (196, 0, 13)
BLUE = (0.12, 0.24, 1)  # (31, 61, 255)
GREEN = (0.122, 0.816, 0.51)  # (31, 208, 130)
NAVYBLUE = (0, 0, 0.4)  # (0, 0, 102)
BLACK = (0, 0, 0)
WHITE = (1, 1, 1)
CGREEN = (0.57254902, 0.7254902, 0.51372549)  # (146, 185, 131)
CBLUE = (0.70196078, 0.83137255, 1)  # (179, 212, 255)

# Top domain value (to be understood)
TOP_DOMAIN = 53.32  # 90%

# Main datafile
DATAFILE = "./data/2020_data.csv"


def import_consts(start_date="2020-01-01", end_date="2022-01-01", negative_prices=False):
    """
    Import data and set constants for the analysis.

    Args:
        start_date (str): Start date of the analysis.
        end_date (str): End date of the analysis.
        negative_prices (bool): If True, allows for negative prices.

    Returns:
        tuple: Various datasets and constants used in the analysis.
    """
    all_data = pd.read_csv(DATAFILE)
    all_data.index = pd.date_range(start="2020-01-01", periods=len(all_data), freq="h")
    all_data = all_data.loc[start_date:end_date]
    prices_B = np.maximum(all_data["UP"].to_numpy(), 0)
    prices_S = np.maximum(all_data["DW"].to_numpy(), 0)
    prices_F = np.maximum(all_data["forward_RE"].to_numpy(), 0)
    prices_forecast = np.maximum(all_data["forward_FC"].to_numpy(), 0)

    features = all_data.loc[:, ["Offshore DK2", "Offshore DK1", "Onshore DK2", "Onshore DK1", "production_FC"]]
    features["forward"] = prices_F
    features_red = all_data.loc[:, ["production_FC"]]
    features_red["forward"] = prices_F
    realized = all_data.loc[:, "production_RE"].to_numpy()
    realized *= NOMINAL_WIND

    fm_df = pd.read_csv("./results/2020_forecast_model.csv")
    forecast_production = fm_df.loc[:, "forecast_production"]

    penalty = np.quantile(prices_B, 0.95)

    return (
        prices_B, prices_S, prices_F, prices_forecast, features,
        features_red, realized, PRICE_H, P_H, NOMINAL_WIND,
        penalty, H_MIN, forecast_production
    )