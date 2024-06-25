import copy

from abc import ABC, abstractmethod

import gurobipy as gp
from gurobipy import GRB

from DataLoader import DataLoader
from constants import *
from real_time_adj_original import get_hydro_opt
from real_time_adjustment_utils import compute_objective_variable_bids, compute_objective_fixed_bids


class BaseModel(ABC):
    def __init__(self, filename, datafile=DATAFILE, nominal_wind=NOMINAL_WIND, max_wind=NOMINAL_WIND, p_h_max=P_H_MAX,
                 h_min=H_MIN):
        self.data_loader = DataLoader(datafile)
        self.nominal_wind = nominal_wind
        self.max_wind = max_wind
        self.p_h_max = p_h_max
        self.h_min = h_min
        self.M = max(self.max_wind, self.p_h_max) + 9999999

        self._load_and_process_data()
        self.filename = filename
        self.weights = None
        self.trained = False
        self.results = None
        self.evaluated = False

        if os.path.exists(os.path.join(RESULTS_DIR, f"{self.filename}.csv")):
            self.load_weights()
            self.trained = True

    def _load_and_process_data(self):
        data = self.data_loader.get_data()
        prices_F = data["forward_RE"].to_numpy()
        self.x = data[["Offshore DK2", "Offshore DK1", "Onshore DK2", "Onshore DK1", "production_FC", "forward_RE"]]
        self.n_features = self.x.shape[1]
        self.x_fm = [(sum(
            Q_FORECAST_CALCULATED[i] * self.x.iloc[t, i] for i in range(self.n_features - 1)) + Q_INTERCEPT_CALCULATED,
                      prices_F[t]) for t in range(len(prices_F))]

        prices_B, prices_S, prices_F, prices_forecast, features, features_red, realized, forecasted_prod = self.data_loader.build_all(
            self.nominal_wind)
        self.prices_B = prices_B
        self.prices_S = prices_S
        self.prices_F = prices_F
        self.prices_forecast = prices_forecast
        self.realized = realized
        self.forecasted_prod = forecasted_prod
        self.rolling_forecasts = self.data_loader.load_rolling_forecasts(self.nominal_wind)

    def save_weights(self, weights):
        weights.to_csv(os.path.join(RESULTS_DIR, f"{self.filename}.csv"), index=False)
        print("Weights saved successfully")

    def load_weights(self):
        filepath = os.path.join(RESULTS_DIR, f"{self.filename}.csv")
        if not os.path.exists(filepath):
            raise FileNotFoundError("No saved weights found")
        self.weights = pd.read_csv(filepath)
        self.trained = True
        print("Weights loaded successfully")

    def summary(self):
        print(f"Filename: {self.filename}")
        print(f"Nominal wind: {self.nominal_wind}")
        print(f"Max wind: {self.max_wind}")
        print(f"Max hydrogen: {self.p_h_max}")
        print(f"H min: {self.h_min}")
        print(f"Number of features: {self.n_features}")
        print(f"Model trained: {self.trained}")
        print(f"Model evaluated: {self.evaluated}")
        if self.evaluated:
            print(f"Total objective: {self.get_total_objective()}")

    # @abstractmethod
    # def get_schedule(self, idx_start, idx_end):
    #     pass
    #
    # def evaluate(self, idx_start, idx_end, start_from_second_year=False):
    #     """
    #     Evaluate the model on the test set.
    #     Args:
    #         start_from_second_year: If True, the model will start from the second year of data
    #         idx_start: Start index of the data
    #         idx_end: End index of the data
    #
    #     Returns:
    #         dict: Results of the optimization problem
    #     """
    #     if not self.trained:
    #         print("Model has not been trained.")
    #         return
    #
    #     forward_bids, h_prods = self.get_schedule(idx_start, idx_end)
    #
    #     ds, ups, dws, objs, missing_productions = [], [], [], [], []
    #
    #     for t in range(idx_start, idx_end):
    #         h_adj = h_prods[t - idx_start]
    #         forward_bid = forward_bids[t - idx_start]
    #         if t % HOURS_PER_DAY == 0 and t != idx_start:
    #             missing_production = np.maximum(
    #                 self.h_min - np.sum(h_prods[(t - idx_start - HOURS_PER_DAY):(t - idx_start)]), 0)
    #         else:
    #             missing_production = 0
    #         settlement = self.realized[t] - forward_bid - h_adj
    #
    #         up_val = np.maximum(-settlement, 0)
    #         dw_val = np.maximum(settlement, 0)
    #         obj_val = (
    #                 forward_bid * self.prices_F[t]
    #                 + PRICE_H * h_adj
    #                 + dw_val * self.prices_S[t]
    #                 - up_val * self.prices_B[t]
    #                 - missing_production * PENALTY
    #         )
    #
    #         ds.append(self.realized[t] - forward_bid)
    #         ups.append(up_val)
    #         dws.append(dw_val)
    #         missing_productions.append(missing_production)
    #         objs.append(obj_val)
    #
    #     results = {'d': ds, 'h_prod': h_prods, 'forward_bid': forward_bids, 'up': ups, 'dw': dws,
    #                'missing_production': missing_productions, 'obj': objs}
    #
    #     self.results = results
    #     self.evaluated = True
    #
    #     return results

    def get_results(self):
        if not self.evaluated:
            print("Model not evaluated. Please evaluate the model first.")
            return
        return self.results

    def get_total_objective(self):
        if not self.evaluated:
            print("Model not evaluated. Please evaluate the model first.")
            return
        return np.sum(self.results["obj"])

    # Real-Time Adjustment Methods #

    def best_adjustment(self, idx_start, idx_end):

        if not self.evaluated:
            print("Model not evaluated. Please evaluate the model first.")
            return
        results = copy.deepcopy(self.results)

        ds, h_prods, ups, dws, objs, missing_productions = [], [], [], [], [], []

        forward_bids = results['forward_bid']

        h_prod = []

        for t in range(idx_start, idx_end, HOURS_PER_DAY):
            # if (t//HOURS_PER_DAY)%10 == 0:
            #     print(f"Day {t // HOURS_PER_DAY + 1} of {idx_end // HOURS_PER_DAY}")
            # For each day, solve the optimization problem with full knowledge of the future
            model = gp.Model('Hindsight')

            # Variables
            p_adj = model.addMVar(shape=HOURS_PER_DAY, vtype=GRB.CONTINUOUS, name='p_adj', lb=0.0, ub=self.p_h_max)
            up = model.addMVar(shape=HOURS_PER_DAY, vtype=GRB.CONTINUOUS, name='up', lb=0.0)
            dw = model.addMVar(shape=HOURS_PER_DAY, vtype=GRB.CONTINUOUS, name='dw', lb=0.0)
            up_aux = model.addMVar(shape=HOURS_PER_DAY, vtype=GRB.CONTINUOUS, name='up_aux', lb=-GRB.INFINITY)
            dw_aux = model.addMVar(shape=HOURS_PER_DAY, vtype=GRB.CONTINUOUS, name='dw_aux', lb=-GRB.INFINITY)

            # Objective
            model.setObjective(
                compute_objective_fixed_bids(t, idx_start, HOURS_PER_DAY, p_adj, up, dw, forward_bids, self.prices_F,
                                             self.prices_S, self.prices_B),
                GRB.MAXIMIZE)

            # Constraints
            model.addConstr(self.h_min <= p_adj.sum(), 'Daily Production')

            for j in range(HOURS_PER_DAY):
                k = t + j
                settlement = self.realized[k] - forward_bids[k - idx_start] - p_adj[j]
                model.addConstr(up_aux[j] == -settlement, f'up_aux_{j}')
                model.addConstr(dw_aux[j] == settlement, f'dw_aux_{j}')
                model.addGenConstrMax(up[j], [up_aux[j]], constant=0., name=f'up_{j}')
                model.addGenConstrMax(dw[j], [dw_aux[j]], constant=0., name=f'dw_{j}')

            model.setParam('OutputFlag', 0)
            model.optimize()

            if model.status != GRB.OPTIMAL:
                print(f"Optimization failed at {t}")
                break

            for j in range(HOURS_PER_DAY):
                h_prod.append(p_adj[j].X)

        for t in range(idx_start, idx_end):
            h_adj = h_prod[t - idx_start]
            forward_bid = forward_bids[t - idx_start]
            if t % HOURS_PER_DAY == 0 and t != idx_start:
                missing_production = np.maximum(
                    self.h_min - np.sum(h_prod[(t - idx_start - HOURS_PER_DAY):(t - idx_start)]), 0)
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
            h_prods.append(h_adj)
            ups.append(up_val)
            dws.append(dw_val)
            missing_productions.append(missing_production)
            objs.append(obj_val)

        results['d'] = ds
        results['h_prod'] = h_prods
        results['up'] = ups
        results['dw'] = dws
        results['missing_production'] = missing_productions
        results['obj'] = objs

        return results

    def apply_up_and_dw_adj(self, idx_start, idx_end, printing=False):
        """
        Perform complete evaluation for upwards and downwards adjustment performed on a given model.

        Args:
            idx_start (int): Start index for evaluation.
            idx_end (int): End index for evaluation.
            printing (bool, optional): If True, prints adjustments. Defaults to False.

        Returns:
            dict: Results of the adjustment.
        """
        results = copy.deepcopy(self.results)

        min_production = self.h_min

        forward_bids = []
        ds = []
        h_prods = []
        ups = []
        dws = []
        objs = []
        missing_productions = []
        missing_production = 0
        daily_count = 0

        for i, t in enumerate(range(idx_start, idx_end)):

            print(f"i={i}, t={t}")
            print(f"Prices: {self.prices_B[t]}, {self.prices_S[t]}, {self.prices_F[t]}")
            print(f"Realized: {self.realized[t]}")

            hour_of_day = (i % 24)
            if (hour_of_day == 0) and t != idx_start:
                missing_production = np.maximum(min_production - daily_count, 0)
                daily_count = 0

            forward_bid = results['forward_bid'][i]
            h_prod = results['h_prod'][i]

            d = self.realized[t] - forward_bid

            opt_h = get_hydro_opt(d, self.prices_S[t], self.prices_B[t], self.p_h_max)

            if opt_h > h_prod:
                if printing:
                    print(f"i={i}, method 1: We changed from {h_prod} to {opt_h}")
                h_adj = opt_h
            else:
                remaining_hours = 23 - hour_of_day
                if hour_of_day == 23:
                    remaining_planned = 0
                else:
                    remaining_planned = np.sum([results['h_prod'][i + j] for j in range(remaining_hours + 1)])
                surplus = daily_count + remaining_planned - min_production
                wanted = h_prod - opt_h
                if surplus >= wanted:
                    h_adj = opt_h
                else:
                    h_adj = np.minimum(np.maximum(h_prod - surplus, 0), self.p_h_max)
                    if printing:
                        print(f"i={i}, t={t}, hod={hour_of_day}")
                        print(f"planned={remaining_planned}, produced={daily_count}")
                        print(f"surplus={surplus}, wanted={wanted}")
                        print(f"Original prod: {h_prod}, Resulting prod: {h_adj}")

            daily_count += h_adj
            settlementd = self.realized[t] - forward_bid - h_adj
            up = np.maximum(-settlementd, 0)
            dw = np.maximum(settlementd, 0)
            obj = (
                    forward_bid * self.prices_F[t]
                    + PRICE_H * h_adj
                    + dw * self.prices_S[t]
                    - up * self.prices_B[t]
                    - missing_production * PENALTY
            )

            forward_bids.append(forward_bid)
            ds.append(d)
            h_prods.append(h_adj)
            ups.append(up)
            dws.append(dw)
            missing_productions.append(missing_production)
            missing_production = 0
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
        return results

    def MPC_adjustment(self, idx_start, idx_end, verbose=False):
        results = copy.deepcopy(self.results)

        forward_bids = results['forward_bid']

        ds, h_prods, ups, dws, objs, missing_productions = [], [], [], [], [], []
        missing_production = daily_count = 0

        for t in range(idx_start, idx_end):
            i = t % 24
            if i == 0 and t != idx_start:
                if verbose:
                    print(f"Day {t // 24} of {(idx_end - idx_start) // 24} done")
                missing_production = max(self.h_min - daily_count, 0)
                daily_count = 0

            forward_bid = forward_bids[t - idx_start]
            d = self.realized[t] - forward_bid
            hours_left = 24 - i

            if daily_count < self.h_min:
                model = gp.Model('Real Time Adjustment')

                # Variables
                p_adj = model.addMVar(shape=hours_left, vtype=GRB.CONTINUOUS, name='p_adj', lb=0.0, ub=self.p_h_max)
                up = model.addMVar(shape=hours_left, vtype=GRB.CONTINUOUS, name='up', lb=0.0)
                dw = model.addMVar(shape=hours_left, vtype=GRB.CONTINUOUS, name='dw', lb=0.0)
                up_aux = model.addMVar(shape=hours_left, vtype=GRB.CONTINUOUS, name='up_aux', lb=-GRB.INFINITY)
                dw_aux = model.addMVar(shape=hours_left, vtype=GRB.CONTINUOUS, name='dw_aux', lb=-GRB.INFINITY)

                # Objective
                model.setObjective(
                    compute_objective_fixed_bids(t, idx_start, hours_left, p_adj, up, dw, forward_bids, self.prices_F,
                                                 self.prices_S,
                                                 self.prices_B), GRB.MAXIMIZE)

                # Constraints
                model.addConstr(self.h_min <= p_adj.sum() + daily_count, 'Daily Production')

                for j in range(hours_left):
                    k = t + j
                    settlement = self.realized[k] - forward_bids[k - idx_start] - p_adj[j] if j == 0 else \
                        self.rolling_forecasts[t][j] - \
                        forward_bids[k - idx_start] - p_adj[j]
                    model.addConstr(up_aux[j] == -settlement, f'up_aux_{j}')
                    model.addConstr(dw_aux[j] == settlement, f'dw_aux_{j}')
                    model.addGenConstrMax(up[j], [up_aux[j]], constant=0., name=f'up_{j}')
                    model.addGenConstrMax(dw[j], [dw_aux[j]], constant=0, name=f'dw_{j}')

                model.setParam('OutputFlag', 0)
                model.optimize()

                if model.status != GRB.OPTIMAL:
                    if verbose:
                        print(f"Optimization failed at {t}")
                    break

                h_adj = np.maximum(0, np.minimum(self.p_h_max, p_adj[0].X))

            else:
                h_adj = get_hydro_opt(self.realized[t] - forward_bid, self.prices_S[t], self.prices_B[t], self.p_h_max)

            daily_count += h_adj
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

            ds.append(d)
            h_prods.append(h_adj)
            ups.append(up_val)
            dws.append(dw_val)
            missing_productions.append(missing_production)
            objs.append(obj_val)
            missing_production = 0

            if verbose:
                print(f"Time {t}:")
                print(f"  Prices: {self.prices_B[t]}, {self.prices_S[t]}, {self.prices_F[t]}")
                print(f"  Forward Bid: {forward_bids[t]}")
                print(f"  Realized: {self.realized[t]}")
                print(f"  Adjustment: {h_adj}")
                print(f"  Settlement: {settlement}")
                print(f"  Objective Value: {obj_val}")

        results['d'] = ds
        results['h_prod'] = h_prods
        results['up'] = ups
        results['dw'] = dws
        results['missing_production'] = missing_productions
        results['obj'] = objs

        return results
