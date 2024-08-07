import copy
import json
from abc import ABC, abstractmethod

import numpy as np
from matplotlib import pyplot as plt

from utils.DataLoader import DataLoader
from utils.Result import Result
from utils.real_time_adjustment_utils import *
import gurobipy as gp
from gurobipy import GRB

from utils.stochastic_optimization import generate_scenario


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

        prices_SB, prices_F, prices_forecast, features, features_red, realized, forecasted_prod = self.data_loader.build_all(
            self.nominal_wind)
        self.single_balancing_prices = prices_SB
        self.prices_F = prices_F
        self.prices_forecast = prices_forecast
        self.realized = realized
        self.forecasted_prod = forecasted_prod
        self.rolling_forecasts = self.data_loader.load_rolling_forecasts(self.nominal_wind)

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
        idx_end = idx_start + len(results.forward_bids)

        deviations, h_prods, settlements_list, objectives, missing_productions = [], [], [], [], []

        forward_bids = results.forward_bids

        h_prod = []

        for t in range(idx_start, idx_end, HOURS_PER_DAY):
            # if (t//HOURS_PER_DAY)%10 == 0:
            #     print(f"Day {t // HOURS_PER_DAY + 1} of {idx_end // HOURS_PER_DAY}")
            # For each day, solve the optimization problem with full knowledge of the future
            model = gp.Model('Hindsight')

            # Variables
            p_adj = model.addVars(HOURS_PER_DAY, vtype=GRB.CONTINUOUS, name='p_adj', lb=0.0, ub=self.p_h_max)
            settlements = model.addVars(HOURS_PER_DAY, vtype=GRB.CONTINUOUS, name='settlements',
                                        lb=-GRB.INFINITY, ub=GRB.INFINITY)

            # Objective
            model.setObjective(
                compute_objective_fixed_bids(t, idx_start, HOURS_PER_DAY, p_adj, settlements, forward_bids,
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

    def rule_based_adjustment(self, results, info_on_current_hour=True, printing=False):
        """
        Perform complete evaluation for upwards and downwards adjustment performed on a given model.

        Args:
            info_on_current_hour: If True, the model has information on the current hour. Defaults to True.
            results: Results of the model.
            printing (bool, optional): If True, prints adjustments. Defaults to False.

        Returns:
            dict: Results of the adjustment.
        """
        results = copy.deepcopy(results)

        idx_start = self.test_start_index
        idx_end = idx_start + len(results.forward_bids)

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

            forward_bid = results.forward_bids[i]
            h_prod = results.hydrogen_productions[i]

            if info_on_current_hour:
                opt_h = get_hydro_opt(self.single_balancing_prices[t], self.p_h_max)
            else:
                opt_h = get_hydro_opt(self.single_balancing_prices[t - 1], self.p_h_max)

            if opt_h > h_prod:
                if printing:
                    print(f"i={i}, method 1: We changed from {h_prod} to {opt_h}")
                h_adj = opt_h
            else:
                remaining_hours = HOURS_PER_DAY - hour_of_day
                if hour_of_day == HOURS_PER_DAY:
                    remaining_planned = 0
                else:
                    remaining_planned = np.sum(
                        [results.hydrogen_productions[i + j] for j in range(remaining_hours)])
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
            deviations.append(self.realized[t] - forward_bid)
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
        idx_end = idx_start + len(results.forward_bids)

        forward_bids = results.forward_bids

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
                p_adj = model.addVars(hours_left, vtype=GRB.CONTINUOUS, name='p_adj', lb=0.0, ub=self.p_h_max)
                settlements = model.addVars(hours_left, vtype=GRB.CONTINUOUS, name='settlements',
                                            lb=-GRB.INFINITY,
                                            ub=GRB.INFINITY)

                # Objective
                model.setObjective(
                    compute_objective_fixed_bids(t, idx_start, hours_left, p_adj, settlements,
                                                 forward_bids, self.prices_F,
                                                 self.single_balancing_prices), GRB.MAXIMIZE)

                # Constraints
                model.addConstr(self.h_min <= p_adj.sum() + daily_count, 'Daily Production')

                model.addConstr(settlements[0] == self.realized[t] - forward_bids[t - idx_start] - p_adj[0],
                                'settlement')
                for j in range(1, hours_left):
                    k = t + j
                    model.addConstr(
                        settlements[j] == self.rolling_forecasts[t][j] - forward_bids[k - idx_start] - p_adj[j],
                        'settlement')

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
                print(f"  Prices: {self.single_balancing_prices[t]}, {self.prices_F[t]}")
                print(f"  Forward Bid: {forward_bids[t]}")
                print(f"  Realized: {self.realized[t]}")
                print(f"  Adjustment: {h_adj}")
                print(f"  Settlement: {settlement}")
                print(f"  Objective Value: {obj_val}")

        results = Result(forward_bids, deviations, h_prods, settlements_list, missing_productions, objectives)

        return results

    def MPC_adjustment_with_naive_balancing_forecasts(self, results, lag, rolling_forecasts=False,
                                                      info_on_current_hour=True,
                                                      verbose=False):
        results = copy.deepcopy(results)

        idx_start = self.test_start_index
        idx_end = idx_start + len(results.forward_bids)

        forward_bids = results.forward_bids

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
                p_adj = model.addVars(hours_left, vtype=GRB.CONTINUOUS, name='p_adj', lb=0.0, ub=self.p_h_max)
                settlements = model.addVars(hours_left, vtype=GRB.CONTINUOUS, name='settlements',
                                            lb=-GRB.INFINITY,
                                            ub=GRB.INFINITY)

                if rolling_forecasts:
                    model.setObjective(
                        compute_objective_fixed_bids_naive_balancing_prices(t, idx_start, hours_left, p_adj,
                                                                            settlements,
                                                                            forward_bids, self.prices_F,
                                                                            self.single_balancing_prices,
                                                                            lag=hours_left,
                                                                            info_on_current_hour=info_on_current_hour),
                        GRB.MAXIMIZE)
                else:
                    model.setObjective(
                        compute_objective_fixed_bids_naive_balancing_prices(t, idx_start, hours_left, p_adj,
                                                                            settlements,
                                                                            forward_bids, self.prices_F,
                                                                            self.single_balancing_prices, lag,
                                                                            info_on_current_hour=info_on_current_hour),
                        GRB.MAXIMIZE)

                # Constraints
                model.addConstr(self.h_min <= p_adj.sum() + daily_count, 'Daily Production')

                if info_on_current_hour:
                    model.addConstr(settlements[0] == self.realized[t] - forward_bids[t - idx_start] - p_adj[0],
                                    'settlement')
                else:
                    model.addConstr(settlements[0] == self.forecasted_prod[t] - forward_bids[t - idx_start] - p_adj[0],
                                    'settlement')
                for j in range(1, hours_left):
                    k = t + j
                    model.addConstr(
                        settlements[j] == self.forecasted_prod[k] - forward_bids[k - idx_start] - p_adj[j],
                        'settlement')

                model.setParam('OutputFlag', 0)
                model.optimize()

                if model.status != GRB.OPTIMAL:
                    if verbose:
                        print(f"Optimization failed at {t}")
                    break

                h_adj = np.maximum(0, np.minimum(self.p_h_max, p_adj[0].X))

            else:
                if info_on_current_hour:
                    h_adj = get_hydro_opt(self.single_balancing_prices[t], self.p_h_max)
                else:
                    h_adj = get_hydro_opt(self.single_balancing_prices[t - 1], self.p_h_max)

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
                print(f"  Prices: {self.single_balancing_prices[t]}, {self.prices_F[t]}")
                print(f"  Forward Bid: {forward_bid}")
                print(f"  Realized: {self.realized[t]}")
                print(f"  Adjustment: {h_adj}")
                print(f"  Settlement: {settlement}")
                print(f"  Objective Value: {obj_val}")

        results = Result(forward_bids, deviations, h_prods, settlements_list, missing_productions, objectives)

        return results

    def stochastic_MPC_adjustment(self, results, transition_matrices, arma_model_up, arma_model_dw, epsilon,
                                  prices_diff, balancing_states, verbose=False, num_scenarios=10):

        results = copy.deepcopy(results)

        arma_model_up = copy.deepcopy(arma_model_up)
        arma_model_dw = copy.deepcopy(arma_model_dw)

        idx_start = self.test_start_index
        idx_end = idx_start + len(results.forward_bids)

        forward_bids = results.forward_bids

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

            scenario_balancing_prices = np.zeros((num_scenarios, hours_left))
            initial_state = 0 if self.single_balancing_prices[t - 1] < self.prices_F[t - 1] else (
                2 if self.single_balancing_prices[t - 1] > self.prices_F[t - 1] else 1)
            initial_hour = (t - 1) % 24
            for s in range(num_scenarios):
                scenario_balancing_prices[s] = generate_scenario(hours_left, initial_state, initial_hour,
                                                                 transition_matrices, copy.deepcopy(arma_model_up),
                                                                 copy.deepcopy(arma_model_dw),
                                                                 epsilon, prices_diff[t:t + hours_left],
                                                                 balancing_states[t:t + hours_left])
                scenario_balancing_prices[s] = self.prices_F[t:t + hours_left] - scenario_balancing_prices[s]
            # plt.figure()
            # plt.plot(scenario_balancing_prices[0, :],linestyle='--', label='Scenario 1',color='blue')
            # plt.plot(self.prices_F[t:t + hours_left], label='Forward Prices',color='red')
            # plt.plot(self.single_balancing_prices[t:t + hours_left], linestyle='--', label='Real balancing prices',color='green')
            # plt.legend()
            # plt.show()

            if daily_count < self.h_min:
                # Generate scenarios

                model = gp.Model('Stochastic Real Time Adjustment')

                # Variables
                p_adj = model.addVars(hours_left, vtype=GRB.CONTINUOUS, name='p_adj', lb=0.0, ub=self.p_h_max)
                settlements = model.addVars(hours_left, vtype=GRB.CONTINUOUS, name='settlements',
                                            lb=-GRB.INFINITY, ub=GRB.INFINITY)

                p_adj = np.array([p_adj[j] for j in range(hours_left)])
                settlements = np.array([settlements[j] for j in range(hours_left)])

                # Set the objective to maximize the expected objective across scenarios
                model.setObjective(
                    compute_objective_fixed_bids_balancing_prices_forecasts_scenarios(hours_left, p_adj,
                                                                                      settlements,
                                                                                      scenario_balancing_prices),
                    GRB.MAXIMIZE)

                # Constraints
                model.addConstr(self.h_min <= p_adj.sum() + daily_count, 'Daily Production')

                for s in range(num_scenarios):
                    for j in range(hours_left):
                        k = t + j
                        model.addConstr(
                            settlements[s, j] == self.forecasted_prod[k] - forward_bids[k - idx_start] - p_adj[j],
                            f'settlement_scenario_{s, j}')

                model.setParam('OutputFlag', 0)
                model.optimize()

                if model.status != GRB.OPTIMAL:
                    if verbose:
                        print(f"Optimization failed at {t}")
                    break

                h_adj = np.maximum(0, np.minimum(self.p_h_max, p_adj[0].X))

            else:
                expected_balancing_price = np.mean(scenario_balancing_prices[:, 0])
                h_adj = get_hydro_opt(expected_balancing_price, self.p_h_max)

            if balancing_states[t] == 0:
                arma_model_dw.update(np.log(prices_diff[t] + epsilon))
            elif balancing_states[t] == 2:
                arma_model_up.update(np.log(-prices_diff[t] + epsilon))

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
                print(f"  Prices: {self.single_balancing_prices[t]}, {self.prices_F[t]}")
                print(f"  Forward Bid: {forward_bid}")
                print(f"  Realized: {self.realized[t]}")
                print(f"  Adjustment: {h_adj}")
                print(f"  Settlement: {settlement}")
                print(f"  Objective Value: {obj_val}")

        results = Result(forward_bids, deviations, h_prods, settlements_list, missing_productions, objectives)

        return results

    def stochastic_MPC_adjustment_load_scenarios(self, results,
                                                 scenarios_file='../results/stochastic_optimization/100_balancing_prices_scenarios_year.npy',
                                                 verbose=False, num_scenarios=10):

        results = copy.deepcopy(results)

        scenarios = np.load(scenarios_file)

        idx_start = self.test_start_index
        idx_end = idx_start + len(results.forward_bids)

        forward_bids = results.forward_bids

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

            # Sample s scenarios without replacement
            scenarios_indices = np.random.choice(scenarios.shape[1], num_scenarios, replace=False)
            balancing_prices_scenarios = scenarios[t - idx_start, scenarios_indices, :]

            if daily_count < self.h_min:
                # Generate scenarios

                model = gp.Model('Robust Real Time Adjustment')

                # Variables
                p_adj = model.addVars(hours_left, vtype=GRB.CONTINUOUS, name='p_adj', lb=0.0, ub=self.p_h_max)
                settlements = model.addVars(hours_left, vtype=GRB.CONTINUOUS, name='settlements',
                                            lb=-GRB.INFINITY, ub=GRB.INFINITY)

                p_adj = np.array([p_adj[j] for j in range(hours_left)])
                settlements = np.array([settlements[j] for j in range(hours_left)])

                # Set the objective to maximize the expected objective across scenarios
                model.setObjective(
                    compute_objective_fixed_bids_balancing_prices_forecasts_scenarios(hours_left, p_adj,
                                                                                      settlements,
                                                                                      balancing_prices_scenarios),
                    GRB.MAXIMIZE)

                # Constraints
                model.addConstr(self.h_min <= p_adj.sum() + daily_count, 'Daily Production')

                for j in range(hours_left):
                    k = t + j
                    model.addConstr(
                        settlements[j] == self.forecasted_prod[k] - forward_bids[k - idx_start] - p_adj[j],
                        f'settlement_scenario_{j}')

                model.setParam('OutputFlag', 0)
                model.optimize()

                if model.status != GRB.OPTIMAL:
                    if verbose:
                        print(f"Optimization failed at {t}")
                    break

                h_adj = np.maximum(0, np.minimum(self.p_h_max, p_adj[0].X))

            else:
                expected_balancing_price = np.mean(balancing_prices_scenarios[:, 0])
                h_adj = get_hydro_opt(expected_balancing_price, self.p_h_max)

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
                print(f"  Prices: {self.single_balancing_prices[t]}, {self.prices_F[t]}")
                print(f"  Forward Bid: {forward_bid}")
                print(f"  Realized: {self.realized[t]}")
                print(f"  Adjustment: {h_adj}")
                print(f"  Settlement: {settlement}")
                print(f"  Objective Value: {obj_val}")

        results = Result(forward_bids, deviations, h_prods, settlements_list, missing_productions, objectives)

        return results

    def stochastic_MPC_adjustment_load_scenarios_worst_case_scenario(self, results,
                                                                     scenarios_file='../results/stochastic_optimization/100_balancing_prices_scenarios_year.npy',
                                                                     verbose=False, num_scenarios=10):

        results = copy.deepcopy(results)

        scenarios = np.load(scenarios_file)

        idx_start = self.test_start_index
        idx_end = idx_start + len(results.forward_bids)

        forward_bids = results.forward_bids

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

            # Sample s scenarios without replacement
            scenario_indices = np.random.choice(scenarios.shape[1], num_scenarios, replace=False)
            balancing_prices_scenarios = scenarios[t - idx_start, scenario_indices, :]

            if daily_count < self.h_min:
                # Generate scenarios

                model = gp.Model('Robust Real Time Adjustment')

                # Variables
                p_adj = model.addVars(hours_left, vtype=GRB.CONTINUOUS, name='p_adj', lb=0.0, ub=self.p_h_max)
                settlements = model.addVars(hours_left, vtype=GRB.CONTINUOUS, name='settlements',
                                            lb=-GRB.INFINITY, ub=GRB.INFINITY)
                min_objective = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name='min_objective')

                p_adj = np.array([p_adj[j] for j in range(hours_left)])
                settlements = np.array([settlements[j] for j in range(hours_left)])

                # Set the objective to maximize the expected objective across scenarios
                model.setObjective(min_objective, GRB.MAXIMIZE)

                # Constraints
                model.addConstr(self.h_min <= p_adj.sum() + daily_count, 'Daily Production')

                for j in range(hours_left):
                    k = t + j
                    model.addConstr(
                        settlements[j] == self.forecasted_prod[k] - forward_bids[k - idx_start] - p_adj[j],
                        f'settlement_scenario_{j}')

                for s in range(num_scenarios):
                    model.addConstr(
                        min_objective <= compute_objective_fixed_bids_balancing_prices_forecasts_single_scenario(hours_left,
                                                                                                           p_adj,
                                                                                                           settlements,
                                                                                                           balancing_prices_scenarios[
                                                                                                               s]))

                model.setParam('OutputFlag', 0)
                model.optimize()

                if model.status != GRB.OPTIMAL:
                    if verbose:
                        print(f"Optimization failed at {t}")
                    break

                h_adj = np.maximum(0, np.minimum(self.p_h_max, p_adj[0].X))

            else:
                expected_balancing_price = np.mean(balancing_prices_scenarios[:, 0])
                h_adj = get_hydro_opt(expected_balancing_price, self.p_h_max)

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
                print(f"  Prices: {self.single_balancing_prices[t]}, {self.prices_F[t]}")
                print(f"  Forward Bid: {forward_bid}")
                print(f"  Realized: {self.realized[t]}")
                print(f"  Adjustment: {h_adj}")
                print(f"  Settlement: {settlement}")
                print(f"  Objective Value: {obj_val}")

        results = Result(forward_bids, deviations, h_prods, settlements_list, missing_productions, objectives)

        return results
