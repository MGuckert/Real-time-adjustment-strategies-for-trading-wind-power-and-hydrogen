from abc import ABC, abstractmethod

from constants import *
from BaseModel import BaseModel


class TrainableModel(BaseModel, ABC):
    def __init__(self, name, test_start_index, datafile, nominal_wind, max_wind, p_h_max, h_min):
        super().__init__(name, test_start_index, datafile, nominal_wind, max_wind, p_h_max, h_min)
        self.weights = None
        self.trained = False
        self.summary()

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def get_schedule_from_weights(self, idx_end, fm):
        pass

    @abstractmethod
    def evaluate(self, eval_length):
        pass

    def save_weights(self, weights):
        weights.to_csv(os.path.join(self.model_dir, f"{self.name}_weights.csv"), index=False)
        print("Weights successfully saved ")

    def load_from_weights(self):
        filepath = os.path.join(self.model_dir, f"{self.name}_weights.csv")
        if not os.path.exists(filepath):
            raise FileNotFoundError("No weights found")
        self.weights = pd.read_csv(filepath)
        self.trained = True
        print("Weights successfully loaded ")

    def evaluate(self, eval_length, fm=False):
        """
        Perform complete evaluation for hourly models with price domains in a given time period.

        Args:
            eval_length (int): End index.
            fm (bool): Flag for model.

        Returns:
            dict: Results of the evaluation.
        """

        if self.weights is None:
            try:
                self.load_from_weights()
            except FileNotFoundError:
                print("Model not trained. Please train the model, or load existing weights.")
                return

        deviations = []
        ups = []
        dws = []
        objectives = []
        missing_productions = []
        missing_production = 0
        daily_count = 0

        dataframe = self.get_schedule_from_weights(eval_length, fm)
        forward_bids = dataframe["forward_bid"].to_numpy()
        h_prods = dataframe["hydrogen_production"].to_numpy()

        for t in range(self.test_start_index, self.test_start_index + eval_length):
            hour_of_day = (t % 24)
            if hour_of_day == 0 and t != self.test_start_index:
                missing_production = np.maximum(self.h_min - daily_count, 0)
                daily_count = 0

            forward_bid = forward_bids[t - self.test_start_index]
            h_prod = h_prods[t - self.test_start_index]

            d = self.realized[t] - forward_bid

            daily_count += h_prod
            settlement = self.realized[t] - forward_bid - h_prod
            up = np.maximum(-settlement, 0)
            dw = np.maximum(settlement, 0)
            obj = (
                    forward_bid * self.prices_F[t]
                    + PRICE_H * h_prod
                    + dw * self.prices_S[t]
                    - up * self.prices_B[t]
                    - missing_production * PENALTY
            )

            deviations.append(d)
            ups.append(up)
            dws.append(dw)
            missing_productions.append(missing_production)
            missing_production = 0
            objectives.append(obj)

        results = {
            "forward_bids": forward_bids,
            "deviations": deviations,
            "hydrogen_productions": h_prods,
            "up": ups,
            "dw": dws,
            "missing_productions": missing_productions,
            "objectives": objectives,
        }

        return results