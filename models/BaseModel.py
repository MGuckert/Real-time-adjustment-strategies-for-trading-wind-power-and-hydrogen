import copy
import json
from abc import ABC

from utils.DataLoader import DataLoader
from utils.Result import Result
from utils.real_time_adjustment_utils import *


class BaseModel(ABC):
    def __init__(self, name, test_start_index, datafile, nominal_wind, max_wind, p_h_max, h_min):
        self.name = name
        self.data_loader = DataLoader(datafile)
        self.nominal_wind = nominal_wind
        self.max_wind = max_wind
        self.p_h_max = p_h_max
        self.h_min = h_min
        self.M = M
        self.test_start_index = test_start_index
        self.model_dir = os.path.join(RESULTS_DIR, self.name)

        # Generate a configuration file for the model
        config = {
            "name": name,
            "nominal_wind": nominal_wind,
            "max_wind": max_wind,
            "p_h_max": p_h_max,
            "h_min": h_min,
            "test_start_index": test_start_index,
            "datafile": datafile
        }
        if os.path.exists(self.model_dir):
            with open(os.path.join(self.model_dir, "config.json"), "r") as f:
                existing_config = json.load(f)
            if existing_config != config:
                raise ValueError(f"Model {name} already exists with different configuration.")
        else:
            os.makedirs(self.model_dir)
        with open(os.path.join(self.model_dir, "config.json"), "w") as f:
            json.dump(config, f)
        self._load_and_process_data()

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
        self.single_balancing_prices = self.data_loader.build_single_balancing_price()

    @staticmethod
    def load_from_config(model_name, model_class):
        model_dir = os.path.join(RESULTS_DIR, model_name)
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"No model found with name {model_name}.")
        with open(os.path.join(model_dir, "config.json"), "r") as f:
            config = json.load(f)
        return model_class(**config)

    def save_results(self, results, flag=""):
        results_dict = results.__dict__
        dataframe = pd.DataFrame(results_dict)
        dataframe.to_csv(os.path.join(self.model_dir, f"{self.name}_results_{flag}.csv"), index=False)
        print("Results saved successfully.")

    def load_results(self, flag=""):
        filepath = os.path.join(self.model_dir, f"{self.name}_results_{flag}.csv")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No results found for {self.name} with flag {flag}.")
        dataframe = pd.read_csv(filepath)
        results = dataframe.to_dict(orient='list')
        results = Result(**results)
        print("Results loaded successfully.")
        return results

    def summary(self):
        print(f"Model {self.name}")
        print(f"Nominal wind: {self.nominal_wind}")
        print(f"Max wind: {self.max_wind}")
        print(f"Max hydrogen: {self.p_h_max}")
        print(f"H min: {self.h_min}")
        print(f"Number of features: {self.n_features}")
        print(f"Test start index: {self.test_start_index}")

    # Real-Time Adjustment Methods #

    def best_adjustment(self, results):

        results = copy.deepcopy(results)

        idx_start = self.test_start_index
        idx_end = idx_start + len(results['forward_bids'])

        deviations, h_prods, settlements_list, objectives, missing_productions = [], [], [], [], []

        forward_bids = results['forward_bids']

        h_prod = []

        for t in range(idx_start, idx_end, HOURS_PER_DAY):
            # if (t//HOURS_PER_DAY)%10 == 0:
            #     print(f"Day {t // HOURS_PER_DAY + 1} of {idx_end // HOURS_PER_DAY}")
            # For each day, solve the optimization problem with full knowledge of the future
            model = gp.Model('Hindsight')

            # Variables
            p_adj = model.addMVar(shape=(HOURS_PER_DAY,), vtype=GRB.CONTINUOUS, name='p_adj', lb=0.0, ub=self.p_h_max)
            settlements = model.addMVar(shape=(HOURS_PER_DAY,), vtype=GRB.CONTINUOUS, name='settlements',
                                        lb=-GRB.INFINITY, ub=GRB.INFINITY)

            # Objective
            model.setObjective(
                compute_objective_single_price_fixed_bids(t, idx_start, HOURS_PER_DAY, p_adj, settlements, forward_bids,
                                                          self.prices_F,
                                                          self.single_balancing_prices),
                GRB.MAXIMIZE)

            # Constraints
            model.addConstr(self.h_min <= p_adj.sum(), 'Daily Production')

            for j in range(HOURS_PER_DAY):
                k = t + j
                model.addConstr(settlements[j] == self.realized[k] - forward_bids[k - idx_start] - p_adj[j],
                                name='settlement')

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

            obj_val = (
                    forward_bid * self.prices_F[t]
                    + PRICE_H * h_adj
                    + settlement * self.single_balancing_prices[t]
                    - missing_production * PENALTY
            )

            deviations.append(self.realized[t] - forward_bid)
            h_prods.append(h_adj)
            settlements_list.append(settlement)
            missing_productions.append(missing_production)
            objectives.append(obj_val)

        results = Result(forward_bids, deviations, h_prods, settlements_list, missing_productions, objectives)

        return results

    def rule_based_adjustment(self, results, printing=False):
        """
        Perform complete evaluation for upwards and downwards adjustment performed on a given model.

        Args:
            results: Results of the model.
            printing (bool, optional): If True, prints adjustments. Defaults to False.

        Returns:
            dict: Results of the adjustment.
        """
        results = copy.deepcopy(results)

        idx_start = self.test_start_index
        idx_end = idx_start + len(results['forward_bids'])

        min_production = self.h_min

        forward_bids = []
        deviations = []
        h_prods = []
        settlements = []
        objectives = []
        missing_productions = []
        missing_production = 0
        daily_count = 0

        for i, t in enumerate(range(idx_start, idx_end)):
            hour_of_day = (i % 24)
            if (hour_of_day == 0) and t != idx_start:
                missing_production = np.maximum(min_production - daily_count, 0)
                daily_count = 0

            forward_bid = results['forward_bids'][i]
            h_prod = results['hydrogen_productions'][i]

            d = self.realized[t] - forward_bid

            opt_h = get_hydro_opt(self.single_balancing_prices[t],self.p_h_max)

            if opt_h > h_prod:
                if printing:
                    print(f"i={i}, method 1: We changed from {h_prod} to {opt_h}")
                h_adj = opt_h
            else:
                remaining_hours = 23 - hour_of_day
                if hour_of_day == 23:
                    remaining_planned = 0
                else:
                    remaining_planned = np.sum(
                        [results['hydrogen_productions'][i + j] for j in range(remaining_hours + 1)])
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
            settlement = self.realized[t] - forward_bid - h_adj
            obj = (
                    forward_bid * self.prices_F[t]
                    + PRICE_H * h_adj
                    + self.single_balancing_prices[t] * settlement
                    - missing_production * PENALTY
            )

            forward_bids.append(forward_bid)
            deviations.append(d)
            h_prods.append(h_adj)
            settlements.append(settlement)
            missing_productions.append(missing_production)
            missing_production = 0
            objectives.append(obj)

        results = Result(forward_bids, deviations, h_prods, settlements, missing_productions, objectives)

        return results

    def MPC_adjustment(self, results, verbose=False):
        results = copy.deepcopy(results)

        idx_start = self.test_start_index
        idx_end = idx_start + len(results['forward_bids'])

        forward_bids = results['forward_bids']

        deviations, h_prods, settlements_list, objectives, missing_productions = [], [], [], [], []
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
                p_adj = model.addMVar(shape=(hours_left,), vtype=GRB.CONTINUOUS, name='p_adj', lb=0.0, ub=self.p_h_max)
                settlements = model.addMVar(shape=(hours_left,), vtype=GRB.CONTINUOUS, name='settlements',
                                            lb=-GRB.INFINITY,
                                            ub=GRB.INFINITY)

                # Objective
                model.setObjective(
                    compute_objective_single_price_fixed_bids(t, idx_start, hours_left, p_adj, settlements,
                                                              forward_bids, self.prices_F,
                                                              self.single_balancing_prices), GRB.MAXIMIZE)

                # Constraints
                model.addConstr(self.h_min <= p_adj.sum() + daily_count, 'Daily Production')

                model.addConstr(settlements[0] == self.realized[t] - forward_bids[t - idx_start] - p_adj[0], 'settlement')
                for j in range(1,hours_left):
                    k = t + j
                    model.addConstr(settlements[j] == self.rolling_forecasts[t][j] - forward_bids[k - idx_start] - p_adj[j], 'settlement')

                model.setParam('OutputFlag', 0)
                model.optimize()

                if model.status != GRB.OPTIMAL:
                    if verbose:
                        print(f"Optimization failed at {t}")
                    break

                h_adj = np.maximum(0, np.minimum(self.p_h_max, p_adj[0].X))

            else:
                h_adj = get_hydro_opt(self.single_balancing_prices[t], self.p_h_max)

            daily_count += h_adj
            settlement = self.realized[t] - forward_bid - h_adj

            obj_val = (
                    forward_bid * self.prices_F[t]
                    + PRICE_H * h_adj
                    + settlement * self.single_balancing_prices[t]
                    - missing_production * PENALTY
            )

            deviations.append(d)
            h_prods.append(h_adj)
            settlements_list.append(settlement)
            missing_productions.append(missing_production)
            objectives.append(obj_val)
            missing_production = 0

            if verbose:
                print(f"Time {t}:")
                print(f"  Prices: {self.prices_B[t]}, {self.prices_S[t]}, {self.prices_F[t]}")
                print(f"  Forward Bid: {forward_bids[t]}")
                print(f"  Realized: {self.realized[t]}")
                print(f"  Adjustment: {h_adj}")
                print(f"  Settlement: {settlement}")
                print(f"  Objective Value: {obj_val}")

        results = Result(forward_bids, deviations, h_prods, settlements_list, missing_productions, objectives)

        return results

    def MPC_adjustment_with_DA_forecasts(self, results, verbose=False):
        day_ahead_forecasts = self.x['production_FC'].to_numpy()

        results = copy.deepcopy(results)

        idx_start = self.test_start_index
        idx_end = idx_start + len(results['forward_bids'])

        forward_bids = results['forward_bids']

        deviations, h_prods, settlements, objectives, missing_productions = [], [], [], [], []
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
                p_adj = model.addMVar(shape=(hours_left,), vtype=GRB.CONTINUOUS, name='p_adj', lb=0.0, ub=self.p_h_max)
                settlements = model.addMVar(shape=(hours_left,), vtype=GRB.CONTINUOUS, name='settlements',
                                            lb=-GRB.INFINITY,
                                            ub=GRB.INFINITY)

                # Objective
                model.setObjective(
                    compute_objective_single_price_fixed_bids(t, idx_start, hours_left, p_adj, settlements,
                                                              forward_bids, self.prices_F,
                                                              self.single_balancing_prices), GRB.MAXIMIZE)

                # Constraints
                model.addConstr(self.h_min <= p_adj.sum() + daily_count, 'Daily Production')

                model.addConstr(settlements[0] == self.realized[t] - forward_bids[t - idx_start] - p_adj[0],
                                'settlement')
                for j in range(1, hours_left):
                    k = t + j
                    model.addConstr(
                        settlements[j] == day_ahead_forecasts[k] - forward_bids[k - idx_start] - p_adj[j],
                        'settlement')

                model.setParam('OutputFlag', 0)
                model.optimize()

                if model.status != GRB.OPTIMAL:
                    if verbose:
                        print(f"Optimization failed at {t}")
                    break

                h_adj = np.maximum(0, np.minimum(self.p_h_max, p_adj[0].X))

            else:
                h_adj = get_hydro_opt(self.single_balancing_prices, self.p_h_max)

            daily_count += h_adj
            settlement = self.realized[t] - forward_bid - h_adj

            obj_val = (
                    forward_bid * self.prices_F[t]
                    + PRICE_H * h_adj
                    + settlement * self.single_balancing_prices[t]
                    - missing_production * PENALTY
            )

            deviations.append(d)
            h_prods.append(h_adj)
            settlements.append(settlement)
            missing_productions.append(missing_production)
            objectives.append(obj_val)
            missing_production = 0

            if verbose:
                print(f"Time {t}:")
                print(f"  Prices: {self.prices_B[t]}, {self.prices_S[t]}, {self.prices_F[t]}")
                print(f"  Forward Bid: {forward_bids[t]}")
                print(f"  Realized: {self.realized[t]}")
                print(f"  Adjustment: {h_adj}")
                print(f"  Settlement: {settlement}")
                print(f"  Objective Value: {obj_val}")

        results = Result(forward_bids, deviations, h_prods, settlements, missing_productions, objectives)

        return results
