import gurobipy as gp
from gurobipy import GRB

from BaseModel import BaseModel
from constants import *
from real_time_adjustment_utils import compute_objective_variable_bids


class HindsightModel(BaseModel):
    def __init__(self, filename, datafile=DATAFILE, nominal_wind=NOMINAL_WIND, max_wind=NOMINAL_WIND, p_h_max=P_H_MAX, h_min=H_MIN):
        super().__init__(filename, datafile, nominal_wind, max_wind, p_h_max, h_min)
        self.summary()

    def compute_hindsight_schedule(self, idx_start=0, idx_end=24*HOURS_PER_MONTH, start_from_second_year=False):

        """
        Solve the optimization problem with full knowledge of the future.
        To test on the "test set" (second year of data), set idx_start=12*HOURS_PER_MONTH, idx_end=test_length + 12*HOURS_PER_MONTH.
        Args:
            start_from_second_year: If True, the model will start from the second year of data
            idx_start: Start index of the data
            idx_end: End index of the data

        Returns:
            dict: Results of the optimization problem

        """

        if start_from_second_year:
            idx_start += 12 * HOURS_PER_MONTH
            idx_end += 12 * HOURS_PER_MONTH

        ds, forward_bids, h_prods, ups, dws, objs, missing_productions = [], [], [], [], [], [], []

        for t in range(idx_start, idx_end, HOURS_PER_DAY):
            # For each day, solve the optimization problem with full knowledge of the future

            model = gp.Model('Global hindsight')

            # Variables
            p_adj = model.addMVar(shape=HOURS_PER_DAY, vtype=GRB.CONTINUOUS, name='p_adj', lb=0.0, ub=self.p_h_max)
            fw_bids = model.addMVar(shape=HOURS_PER_DAY, vtype=GRB.CONTINUOUS, name='forward_bids', lb=-self.p_h_max,
                                    ub=self.nominal_wind)
            up = model.addMVar(shape=HOURS_PER_DAY, vtype=GRB.CONTINUOUS, name='up', lb=0.0)
            dw = model.addMVar(shape=HOURS_PER_DAY, vtype=GRB.CONTINUOUS, name='dw', lb=0.0)
            up_aux = model.addMVar(shape=HOURS_PER_DAY, vtype=GRB.CONTINUOUS, name='up_aux', lb=-GRB.INFINITY)
            dw_aux = model.addMVar(shape=HOURS_PER_DAY, vtype=GRB.CONTINUOUS, name='dw_aux', lb=-GRB.INFINITY)

            # Objective
            model.setObjective(
                compute_objective_variable_bids(t, HOURS_PER_DAY, p_adj, up, dw, fw_bids, self.prices_F, self.prices_S,
                                                self.prices_B),
                GRB.MAXIMIZE)

            # Constraints
            model.addConstr(self.h_min <= p_adj.sum(), 'Daily Production')

            for j in range(HOURS_PER_DAY):
                k = t + j
                settlement = self.realized[k] - fw_bids[j] - p_adj[j]
                model.addConstr(up_aux[j] == -settlement, f'up_aux_{j}')
                model.addConstr(dw_aux[j] == settlement, f'dw_aux_{j}')
                model.addGenConstrMax(up[j], [up_aux[j]], constant=0., name=f'up_{j}')
                model.addGenConstrMax(dw[j], [dw_aux[j]], constant=0., name=f'dw_{j}')

            model.setParam('OutputFlag', 0)
            model.setParam('DualReductions', 0)
            model.optimize()

            if model.status != GRB.OPTIMAL:
                print(f"Optimization failed at {t}")
                break

            for j in range(HOURS_PER_DAY):
                h_prods.append(p_adj[j].X)
                forward_bids.append(fw_bids[j].X)

        dataframe = pd.DataFrame([forward_bids, h_prods]).T
        dataframe.columns = ["forward bid", "hydrogen production"]

        self.save_weights(dataframe)
        self.weights = dataframe
        self.trained = True

    def evaluate(self, idx_start, idx_end, start_from_second_year=False):
        """
        Evaluate the model on the test set.
        Args:
            start_from_second_year: If True, the model will start from the second year of data
            idx_start: Start index of the data
            idx_end: End index of the data

        Returns:
            dict: Results of the optimization problem
        """
        if not self.trained:
            print("Model has not been trained.")
            return

        h_prods, forward_bids = self.weights["hydrogen production"].to_numpy(), self.weights["forward bid"].to_numpy()

        ds, ups, dws, objs, missing_productions = [], [], [], [], []

        for t in range(idx_start, idx_end):
            h_adj = h_prods[t - idx_start]
            forward_bid = forward_bids[t - idx_start]
            if t % HOURS_PER_DAY == 0 and t != idx_start:
                missing_production = np.maximum(
                    self.h_min - np.sum(h_prods[(t - idx_start - HOURS_PER_DAY):(t - idx_start)]), 0)
            else:
                missing_production = 0
            settlement = self.realized[t] - forward_bid - h_adj

            up_val = np.maximum(-settlement, 0)
            dw_val = np.maximum(settlement, 0)
            obj_val = (
                    forward_bid * self.prices_F[t]
                    + PRICE_H * h_adj
                    + dw_val * self.prices_S[t]
                    - up_val * self.prices_B[t]
                    - missing_production * PENALTY
            )

            ds.append(self.realized[t] - forward_bid)
            ups.append(up_val)
            dws.append(dw_val)
            missing_productions.append(missing_production)
            objs.append(obj_val)

        results = {'d': ds, 'h_prod': h_prods, 'forward_bid': forward_bids, 'up': ups, 'dw': dws,
                   'missing_production': missing_productions, 'obj': objs}

        self.results = results
        self.evaluated = True

        return results