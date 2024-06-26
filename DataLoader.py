import pandas as pd

from constants import *


class DataLoader:
    def __init__(self, datafile=DATAFILE):
        self.datafile = datafile
        self._load_data()

    def _load_data(self):
        if not os.path.exists(self.datafile):
            print(f"File {self.datafile} not found.")
            exit(1)
        self.data = pd.read_csv(self.datafile)
        self.data.index = pd.date_range(start="2020-01-01", periods=len(self.data), freq="h")
        self.data = self.data.loc["2020-01-01":"2022-01-01"]

    def get_data(self):
        return self.data

    def build_features(self):
        features = self.data.loc[:, ["Offshore DK2", "Offshore DK1", "Onshore DK2", "Onshore DK1", "production_FC"]]
        features["forward"] = np.maximum(self.data["forward_RE"].to_numpy(), 0)
        return features

    def build_features_red(self):
        features_red = self.data.loc[:, ["production_FC"]]
        features_red["forward"] = np.maximum(self.data["forward_RE"].to_numpy(), 0)
        return features_red

    def build_prices(self):
        prices_B = np.maximum(self.data["UP"].to_numpy(), 0)
        prices_S = np.maximum(self.data["DW"].to_numpy(), 0)
        prices_F = np.maximum(self.data["forward_RE"].to_numpy(), 0)
        prices_forecast = np.maximum(self.data["forward_FC"].to_numpy(), 0)
        return prices_B, prices_S, prices_F, prices_forecast

    def build_production(self):
        realized_prod = self.data["production_RE"].to_numpy()
        forecasted_prod = self.data["production_FC"].to_numpy()
        return realized_prod, forecasted_prod

    def build_all(self, nominal_wind=NOMINAL_WIND):
        prices_B, prices_S, prices_F, prices_forecast = self.build_prices()
        features = self.build_features()
        features_red = self.build_features_red()
        realized_prod, forecasted_prod = self.build_production()
        realized_prod *= nominal_wind
        forecasted_prod *= nominal_wind
        return prices_B, prices_S, prices_F, prices_forecast, features, features_red, realized_prod, forecasted_prod

    @staticmethod
    def load_production_forecasts():
        fm_df = pd.read_csv("./results/2020_forecast_model.csv")
        forecast_production = fm_df.loc[:, "forecast_production"]
        return forecast_production

    @staticmethod
    def load_rolling_forecasts(nominal_wind):
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
        forecasts.clip(lower=0, upper=10, inplace=True)
        forecasts['production_FC'] = forecasts[forecasts.columns].values.tolist()
        return forecasts.loc[:, "production_FC"].to_numpy()
