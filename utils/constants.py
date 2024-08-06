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