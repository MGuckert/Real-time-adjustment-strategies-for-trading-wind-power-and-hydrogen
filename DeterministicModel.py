from BaseModel import BaseModel
from constants import *
from gurobipy import Model, GRB, quicksum
import numpy as np


class DeterministicModel(BaseModel):
    def __init__(self, filename, datafile=DATAFILE, nominal_wind=NOMINAL_WIND, max_wind=NOMINAL_WIND, p_h_max=P_H_MAX, h_min=H_MIN):
        super().__init__(filename, datafile, nominal_wind, max_wind, p_h_max, h_min)
        self.summary()

    def get_deterministic_plan(self, bidding_start=12*HOURS_PER_MONTH):

        offset = bidding_start

        # Declare Gurobi model
        deterministic_plan = Model()

        # Definition of variables
        hydrogen = deterministic_plan.addMVar(shape=HOURS_PER_DAY, lb=0, ub=self.p_h_max, name="hydrogen")
        forward_bids = deterministic_plan.addMVar(shape=HOURS_PER_DAY, lb=-self.p_h_max, ub=self.max_wind,
                                                  name="forward_bid")

        # Maximize profit
        deterministic_plan.setObjective(
            quicksum(
                self.prices_forecast[t + offset] * forward_bids[t]
                + PRICE_H * hydrogen[t]
                for t in range(HOURS_PER_DAY)
            ),
            GRB.MAXIMIZE
        )

        # Min production
        deterministic_plan.addConstr(quicksum(hydrogen[t] for t in range(HOURS_PER_DAY)) >= self.h_min,
                                     name="min_hydrogen")

        # Based on forecasted production
        deterministic_plan.addConstrs(
            (forward_bids[t] + hydrogen[t] == max(0, min(self.forecasted_prod[t + offset], self.max_wind)) for t
             in range(HOURS_PER_DAY)), name="bidding")

        deterministic_plan.setParam('OutputFlag', 0)
        deterministic_plan.setParam('DualReductions', 0)

        deterministic_plan.optimize()

        return hydrogen.X, forward_bids.X

    def generate_plan(self, start, length):
        forward_bids = []
        h_prods = []

        for i in range(start, start + length, 24):
            h, f = self.get_deterministic_plan(i)
            for t in range(24):
                h_prods.append(h[t])
                forward_bids.append(f[t])

        dataframe = pd.DataFrame([forward_bids, h_prods]).T
        dataframe.columns = ["forward bid", "hydrogen production"]

        self.save_weights(dataframe)
        self.weights = dataframe
        self.trained = True

    def evaluate(self, idx_start, idx_end):
        """
            Perform complete evaluation for deterministic with all bids accepted or hindsight models.

            Args:
                forward (list): Forward bids.
                hydrogen (list): Hydrogen schedules.
                idx_start (int): Start index.
                idx_end (int): End index.

            Returns:
                dict: Results of the evaluation.
            """
        schedule = self.weights
        forward_bids = schedule["forward bid"].to_numpy()
        h_prods = schedule["hydrogen production"].to_numpy()

        ds = []
        ups = []
        dws = []
        objs = []
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

            settlementd = self.realized[t] - forward_bid - h_prod
            daily_count += h_prod

            up = np.maximum(-settlementd, 0)
            dw = np.maximum(settlementd, 0)
            obj = (
                    forward_bid * self.prices_F[t]
                    + PRICE_H * h_prod
                    + dw * self.prices_S[t]
                    - up * self.prices_B[t]
                    - missing_production * PENALTY
            )

            ds.append(d)
            ups.append(up)
            missing_productions.append(missing_production)
            missing_production = 0
            dws.append(dw)
            objs.append(obj)

        results = {
            "forward_bid": forward_bids,
            "d": ds,
            "h_prod": h_prods,
            "up": ups,
            "dw": dws,
            "missing_production": missing_productions,
            "obj": objs,
        }

        self.results = results
        self.evaluated = True

        return results


if __name__ == "__main__":
    model = DeterministicModel("test_py", h_min=75)
    model.generate_plan(HOURS_PER_YEAR, HOURS_PER_YEAR)
    model.evaluate(HOURS_PER_YEAR, HOURS_PER_YEAR + 12*HOURS_PER_MONTH)
    print("Done")
