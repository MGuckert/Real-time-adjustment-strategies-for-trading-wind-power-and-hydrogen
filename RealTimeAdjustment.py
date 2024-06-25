import copy
import gurobipy as gp
from gurobipy import GRB, quicksum
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from constants import *
from real_time_adjustment_utils import *
from DataLoader import DataLoader
from BaseModel import BaseModel

class RealTimeAdjustment:
    def __init__(self, datafile=DATAFILE, nominal_wind=NOMINAL_WIND, max_wind=NOMINAL_WIND, p_h_max=P_H_MAX, h_min=H_MIN):
        self.nominal_wind = nominal_wind
        self.max_wind = max_wind
        self.p_h_max = p_h_max
        self.h_min = h_min
        self.data_loader = DataLoader(datafile)
        prices_B, prices_S, prices_F, prices_forecast, features, features_red, realized, forecasted_prod = self.data_loader.build_all(nominal_wind)
        self.prices_B = prices_B[12*HOURS_PER_MONTH:]
        self.prices_S = prices_S[12*HOURS_PER_MONTH:]
        self.prices_F = prices_F[12*HOURS_PER_MONTH:]
        self.realized = realized[12*HOURS_PER_MONTH:]
        self.prod_rolling_forecasts = self.load_rolling_forecasts()

    def load_rolling_forecasts(self, idx_start=0, idx_end=HOURS_PER_YEAR, start_date='01-01-2022', end_date='12-29-2022'):
        forecasts = self.data_loader.load_rolling_forecasts(start_date, end_date)[idx_start:idx_end]
        forecasts *= self.nominal_wind
        forecasts.clip(lower=0, upper=10, inplace=True)
        forecast_cols = [f'FC_{i}h' for i in range(1, 25)]
        forecasts['production_FC'] = forecasts[forecast_cols].values.tolist()
        forecasts.drop(columns=[f'FC_{i}h' for i in range(1, 25)], inplace=True)

        return forecasts
