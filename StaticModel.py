from abc import ABC, abstractmethod

from constants import *
from BaseModel import BaseModel


class StaticModel(BaseModel, ABC):
    def __init__(self, name,test_start_index, datafile, nominal_wind, max_wind, p_h_max, h_min):
        super().__init__(name, test_start_index, datafile, nominal_wind, max_wind, p_h_max, h_min)
        self.summary()

    @abstractmethod
    def get_daily_plan(self, day_index):
        pass

    def generate_schedule(self, length):
        forward_bids = []
        hydrogen_productions = []

        for i in range(self.test_start_index, self.test_start_index + length, 24):
            h, f = self.get_daily_plan(i)
            for t in range(24):
                hydrogen_productions.append(h[t])
                forward_bids.append(f[t])

        dataframe = pd.DataFrame([forward_bids, hydrogen_productions]).T
        dataframe.columns = ["forward_bid", "hydrogen_production"]
        return dataframe

    def evaluate(self, eval_length):
        """
            Perform complete evaluation for deterministic with all bids accepted or hindsight models.

            Args:
                eval_length: Length of the evaluation

            Returns:
                dict: Results of the evaluation.
            """

        schedule = self.generate_schedule(eval_length)

        forward_bids = schedule["forward_bid"].to_numpy()
        h_prods = schedule["hydrogen_production"].to_numpy()

        idx_start = self.test_start_index
        idx_end = idx_start + eval_length

        deviations = []
        ups = []
        dws = []
        objectives = []
        missing_productions = []
        missing_production = 0
        daily_count = 0

        for i, t in enumerate(range(idx_start, idx_end)):
            hour_of_day = (t % 24)
            if hour_of_day == 0 and t != idx_start:
                missing_production = np.maximum(self.h_min - daily_count, 0)
                daily_count = 0

            forward_bid = forward_bids[i]
            d = self.realized[t] - forward_bid
            h_prod = h_prods[i]

            settlement = self.realized[t] - forward_bid - h_prod
            daily_count += h_prod

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
            missing_productions.append(missing_production)
            missing_production = 0
            dws.append(dw)
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
