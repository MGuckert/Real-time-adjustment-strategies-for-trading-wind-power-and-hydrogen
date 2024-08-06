from typing import Any

import pandas as pd

from utils.constants import *


class DataLoader:
    def __init__(self, datafile: str = DATAFILE):
        self.datafile = datafile
        self.data = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        if not os.path.exists(self.datafile):
            print(f"File {self.datafile} not found.")
            exit(1)
        data = pd.read_csv(self.datafile)
        data.index = pd.date_range(start="2020-01-01", periods=len(data), freq="h")
        data = data.loc["2020-01-01":"2022-01-01"]
        return data

    def get_data(self) -> pd.DataFrame:
        return self.data

    def build_features(self) -> pd.DataFrame:
        features = self.data.loc[:, ["Offshore DK2", "Offshore DK1", "Onshore DK2", "Onshore DK1", "production_FC"]]
        features["forward"] = np.maximum(self.data["forward_RE"].to_numpy(), 0)
        return features

    def build_features_red(self) -> pd.DataFrame:
        features_red = self.data.loc[:, ["production_FC"]]
        features_red["forward"] = np.maximum(self.data["forward_RE"].to_numpy(), 0)
        return features_red

    def build_prices(self) -> Any:
        prices_SB = np.maximum(self.data["prices_SB"].to_numpy(), 0)
        prices_F = np.maximum(self.data["forward_RE"].to_numpy(), 0)
        prices_forecast = np.maximum(self.data["forward_FC"].to_numpy(), 0)
        return prices_SB, prices_F, prices_forecast

    def build_production(self) -> Any:
        realized_prod = self.data["production_RE"].to_numpy()
        forecasted_prod = self.data["production_FC"].to_numpy()
        return realized_prod, forecasted_prod

    def build_all(self, nominal_wind: int = NOMINAL_WIND) -> Any:
        prices_SB, prices_F, prices_forecast = self.build_prices()
        features = self.build_features()
        features_red = self.build_features_red()
        realized_prod, forecasted_prod = self.build_production()
        realized_prod *= nominal_wind
        forecasted_prod *= nominal_wind
        return prices_SB, prices_F, prices_forecast, features, features_red, realized_prod, forecasted_prod

    @staticmethod
    def load_rolling_forecasts(nominal_wind: int = NOMINAL_WIND) -> pd.DataFrame:
        """
        Load rolling forecasts from file. WARNING: The first month is not available as it is used for training (mock data has been added)
        Returns:
            pd.DataFrame: Rolling forecasts
        """
        forecasts = pd.read_csv(ROLLING_FORECASTS_FILE, index_col=0)
        # Create a mock first month (720 rows)
        mock_data = np.zeros((720, 24))
        forecasts = pd.concat([pd.DataFrame(mock_data, columns=forecasts.columns), forecasts], ignore_index=True,
                              axis=0)
        forecasts *= nominal_wind
        forecasts.clip(lower=0, upper=nominal_wind, inplace=True)
        forecasts['production_FC'] = forecasts[forecasts.columns].values.tolist()
        return forecasts.loc[:, "production_FC"].to_numpy()
