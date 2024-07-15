from models.StaticModel import StaticModel
from utils.real_time_adjustment_utils import *
import gurobipy as gp
from gurobipy import GRB


class UnconstrainedHindsightModel(StaticModel):
    def __init__(self, name, test_start_index, datafile=DATAFILE, nominal_wind=NOMINAL_WIND, max_wind=NOMINAL_WIND,
                 p_h_max=P_H_MAX, h_min=H_MIN):
        super().__init__(name, test_start_index, datafile, nominal_wind, max_wind, p_h_max, h_min)

    def get_daily_plan(self, day_index):

        model = gp.Model('Global hindsight')

        # Variables
        hydrogen_productions = model.addMVar(shape=(HOURS_PER_DAY,), vtype=GRB.CONTINUOUS, name='p_adj', lb=0.0,
                                             ub=self.p_h_max)
        forward_bids = model.addMVar(shape=(HOURS_PER_DAY,), vtype=GRB.CONTINUOUS, name='forward_bids',
                                     lb=-self.p_h_max,
                                     ub=self.nominal_wind)
        settlements = model.addMVar(shape=(HOURS_PER_DAY,), vtype=GRB.CONTINUOUS, name='settlements', lb=-GRB.INFINITY,
                                    ub=GRB.INFINITY)
        # Objective
        model.setObjective(
            compute_objective_variable_bids(day_index, HOURS_PER_DAY, hydrogen_productions, settlements,
                                            forward_bids, self.prices_F,
                                            self.single_balancing_prices),
            GRB.MAXIMIZE)

        for j in range(HOURS_PER_DAY):
            k = day_index + j
            model.addConstr(settlements[j] == self.realized[k] - forward_bids[j] - hydrogen_productions[j], name='settlement')

        model.setParam('OutputFlag', 0)
        model.setParam('NumericFocus', 1)
        model.optimize()

        if model.status != GRB.OPTIMAL:
            print(f"Optimization failed at {day_index}")

        return hydrogen_productions.X, forward_bids.X