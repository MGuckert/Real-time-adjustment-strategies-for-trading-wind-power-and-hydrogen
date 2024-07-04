from gurobipy import Model, GRB, quicksum

from StaticModel import StaticModel
from utils.constants import *


class DeterministicModel(StaticModel):
    def __init__(self, name, test_start_index, datafile=DATAFILE, nominal_wind=NOMINAL_WIND, max_wind=NOMINAL_WIND,
                 p_h_max=P_H_MAX,
                 h_min=H_MIN):
        super().__init__(name, test_start_index, datafile, nominal_wind, max_wind, p_h_max, h_min)

    def get_daily_plan(self, day_index):

        offset = day_index

        # Declare Gurobi model
        deterministic_plan = Model()

        # Definition of variables
        hydrogen_productions = deterministic_plan.addMVar(shape=HOURS_PER_DAY, lb=0, ub=self.p_h_max, name="hydrogen")
        forward_bids = deterministic_plan.addMVar(shape=HOURS_PER_DAY, lb=-self.p_h_max, ub=self.max_wind,
                                                  name="forward_bids")

        # Maximize profit
        deterministic_plan.setObjective(
            quicksum(
                self.prices_forecast[t + offset] * forward_bids[t]
                + PRICE_H * hydrogen_productions[t]
                for t in range(HOURS_PER_DAY)
            ),
            GRB.MAXIMIZE
        )

        # Min production
        deterministic_plan.addConstr(quicksum(hydrogen_productions[t] for t in range(HOURS_PER_DAY)) >= self.h_min,
                                     name="min_hydrogen")

        # Based on forecasted production
        deterministic_plan.addConstrs(
            (forward_bids[t] + hydrogen_productions[t] == max(0, min(self.forecasted_prod[t + offset], self.max_wind))
             for t
             in range(HOURS_PER_DAY)), name="bidding")

        deterministic_plan.setParam('OutputFlag', 0)
        deterministic_plan.setParam('NumericFocus', 1)

        deterministic_plan.optimize()

        return hydrogen_productions.X, forward_bids.X