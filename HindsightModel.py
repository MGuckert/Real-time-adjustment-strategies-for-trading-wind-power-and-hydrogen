import gurobipy as gp
from gurobipy import GRB

from StaticModel import StaticModel
from constants import *
from real_time_adjustment_utils import compute_objective_variable_bids


class HindsightModel(StaticModel):
    def __init__(self, name, test_start_index, datafile=DATAFILE, nominal_wind=NOMINAL_WIND, max_wind=NOMINAL_WIND,
                 p_h_max=P_H_MAX, h_min=H_MIN):
        super().__init__(name, test_start_index, datafile, nominal_wind, max_wind, p_h_max, h_min)

    def get_daily_plan(self, day_index):

        model = gp.Model('Global hindsight')

        # Variables
        hydrogen_productions = model.addMVar(shape=(HOURS_PER_DAY,), vtype=GRB.CONTINUOUS, name='p_adj', lb=0.0,
                                             ub=self.p_h_max)
        forward_bids = model.addMVar(shape=(HOURS_PER_DAY,), vtype=GRB.CONTINUOUS, name='forward_bids', lb=-self.p_h_max,
                                     ub=self.nominal_wind)
        up = model.addMVar(shape=(HOURS_PER_DAY,), vtype=GRB.CONTINUOUS, name='up', lb=0.0)
        dw = model.addMVar(shape=(HOURS_PER_DAY,), vtype=GRB.CONTINUOUS, name='dw', lb=0.0)
        up_aux = model.addMVar(shape=(HOURS_PER_DAY,), vtype=GRB.CONTINUOUS, name='up_aux', lb=-GRB.INFINITY)
        dw_aux = model.addMVar(shape=(HOURS_PER_DAY,), vtype=GRB.CONTINUOUS, name='dw_aux', lb=-GRB.INFINITY)

        # Objective
        model.setObjective(
            compute_objective_variable_bids(day_index, HOURS_PER_DAY, hydrogen_productions, up, dw, forward_bids, self.prices_F,
                                            self.prices_S,
                                            self.prices_B),
            GRB.MAXIMIZE)

        # Constraints
        model.addConstr(self.h_min <= hydrogen_productions.sum(), 'Daily Production')

        for j in range(HOURS_PER_DAY):
            k = day_index + j
            settlement = self.realized[k] - forward_bids[j] - hydrogen_productions[j]
            model.addConstr(up_aux[j] == -settlement, f'up_aux_{j}')
            model.addConstr(dw_aux[j] == settlement, f'dw_aux_{j}')
            model.addGenConstrMax(up[j], [up_aux[j]], constant=0., name=f'up_{j}')
            model.addGenConstrMax(dw[j], [dw_aux[j]], constant=0., name=f'dw_{j}')

        model.setParam('OutputFlag', 0)
        model.setParam('NumericFocus', 1)
        model.optimize()

        if model.status != GRB.OPTIMAL:
            print(f"Optimization failed at {day_index}")

        return hydrogen_productions.X, forward_bids.X