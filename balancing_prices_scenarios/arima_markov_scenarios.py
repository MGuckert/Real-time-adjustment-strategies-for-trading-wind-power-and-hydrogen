import copy
import os
import time
import warnings
from multiprocessing import Pool
import pickle

import numpy as np
import pandas as pd
from pmdarima import auto_arima
from statsmodels.tools.sm_exceptions import ValueWarning

from models.HAPDModel import HAPDModel
from utils.constants import HOURS_PER_YEAR, HOURS_PER_MONTH, HOURS_PER_DAY, HOURS_PER_WEEK, ORIGINAL
from utils.stochastic_optimization import build_hour_specific_transition_matrices

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ValueWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


def generate_scenario_2(num_steps, initial_state, initial_hour, transition_matrices, arma_model_up,
                        arma_model_dw, epsilon):
    states = np.zeros(num_steps + 1, dtype=int)
    states[0] = initial_state
    hour = initial_hour
    prices_diff = np.empty(HOURS_PER_DAY, dtype=float)
    for i in range(num_steps):
        state = np.random.choice(3, p=transition_matrices[hour][states[i]])
        states[i + 1] = state
        hour = (hour + 1) % 24
    for state in range(3):
        state_indexes = (np.argwhere(states == state) - 1).flatten()
        if len(state_indexes) == 0:
            continue
        prices_diff[state_indexes] = np.exp(
            arma_model_dw.predict(n_periods=len(state_indexes))) - epsilon if state == 0 else (
            -np.exp(arma_model_up.predict(n_periods=len(state_indexes))) + epsilon if state == 2 else 0)
    return prices_diff


def generate_scenario(num_steps, initial_state, initial_hour, transition_matrices, arma_model_up,
                      arma_model_dw, epsilon, prices_diff, balancing_states, forward_prices):
    states = np.zeros(num_steps + 1, dtype=int)
    states[0] = initial_state
    hour = initial_hour
    prices = np.empty(HOURS_PER_DAY, dtype=float)
    for i in range(num_steps):
        forward_price = forward_prices[i]
        state = np.random.choice(3, p=transition_matrices[hour][states[i]])
        states[i + 1] = state
        if state == 0:
            prices[i] = np.exp(arma_model_dw.predict(1)) - epsilon
        elif state == 2:
            prices[i] = -np.exp(arma_model_up.predict(1)) + epsilon
        else:
            prices[i] = 0
        if balancing_states[i] == 0:
            arma_model_dw.update(np.log(prices_diff[i] + epsilon))
        elif balancing_states[i] == 2:
            arma_model_up.update(np.log(-prices_diff[i] + epsilon))
        hour = (hour + 1) % 24
        prices[i] = forward_price - prices[i]
    return prices


def generate_scenarios_whole_year_multiprocessing(model, transition_matrices, arma_model_up, arma_model_dw, epsilon,
                                                  prices_diff, balancing_states, num_scenarios=1):
    idx_start = model.test_start_index
    idx_end = idx_start + HOURS_PER_WEEK

    scenarios = np.empty((idx_end - idx_start, num_scenarios, 24), dtype=float)
    with Pool() as pool:

        initial_hours = (np.arange(idx_start, idx_end) - 1) % 24
        for t in range(idx_start, idx_end):
            i = t % 24
            hours_left = 24 - i

            initial_state = balancing_states[t - 1]
            initial_hour = initial_hours[t - idx_start]

            args = [
                (
                    hours_left, initial_state, initial_hour, transition_matrices, arma_model_up, arma_model_dw,
                    epsilon, prices_diff[t:t + hours_left], balancing_states[t:t + hours_left],
                    model.prices_F[t:t + hours_left]
                )
                for _ in range(num_scenarios)
            ]

            start = time.time()

            results = pool.starmap(generate_scenario, args, chunksize=num_scenarios)
            for j, result in enumerate(results):
                scenarios[t - idx_start][j] = result

            if balancing_states[t] == 0:
                arma_model_dw.update(np.log(prices_diff[t] + epsilon))
            elif balancing_states[t] == 2:
                arma_model_up.update(np.log(-prices_diff[t] + epsilon))

            print("Execution time for hour ", t - idx_start, " is: ", time.time() - start)
    # Save the scenarios
    np.save(f'../results/stochastic_optimization/balancing_prices_scenarios_week1.npy', scenarios)


def generate_scenarios_whole_year_multiprocessing_2(model, transition_matrices, arma_model_up, arma_model_dw, epsilon,
                                                    prices_diff, balancing_states, num_scenarios=1):
    idx_start = model.test_start_index
    idx_end = idx_start + HOURS_PER_YEAR

    scenarios = np.empty((idx_end - idx_start, num_scenarios, 24), dtype=float)
    with Pool() as pool:

        initial_hours = (np.arange(idx_start, idx_end) - 1) % 24
        for t in range(idx_start, idx_end):
            i = t % 24
            hours_left = 24 - i

            initial_state = balancing_states[t - 1]
            initial_hour = initial_hours[t - idx_start]

            args = [
                (
                    hours_left, initial_state, initial_hour, transition_matrices, arma_model_up, arma_model_dw,
                    epsilon
                )
                for _ in range(num_scenarios)
            ]

            start = time.time()

            if t != idx_start and t % HOURS_PER_MONTH == 0:
                # Retrain the ARMA models
                upwards_diff = dataframe['price_diff'][:t][dataframe['balancing_state'] == 2].apply(
                    lambda x: np.log(-x + epsilon))
                downwards_diff = dataframe['price_diff'][:t][dataframe['balancing_state'] == 0].apply(
                    lambda x: np.log(x + epsilon))

                arma_model_up = auto_arima(upwards_diff[-HOURS_PER_MONTH:])
                arma_model_dw = auto_arima(downwards_diff[-HOURS_PER_MONTH:])
                print("Retrained ARMA models at hour ", t - idx_start)
                print("New model orders: ", arma_model_up.order, arma_model_dw.order)

                # Update the transition matrices
                transition_matrices = build_hour_specific_transition_matrices(
                    dataframe['balancing_state'][:t].values,
                    model_order=1)

            results = pool.starmap(generate_scenario_2, args, chunksize=num_scenarios)
            for j, result in enumerate(results):
                scenarios[t - idx_start][j] = model.prices_F[t:t + HOURS_PER_DAY] - result

            if balancing_states[t] == 0:
                arma_model_dw.update(np.log(prices_diff[t] + epsilon))
            elif balancing_states[t] == 2:
                arma_model_up.update(np.log(-prices_diff[t] + epsilon))
    # Save the scenarios
    np.save(f'../results/stochastic_optimization/100_balancing_prices_scenarios_year_improved.npy', scenarios)


if __name__ == '__main__':
    # logging.basicConfig(level=logging.DEBUG)
    # mp.log_to_stderr(logging.DEBUG)
    #
    initial_time = time.time()

    dataframe = pd.read_csv(
        "../data/2022_2023/2022_2023_data.csv")

    balancing_states = np.where(dataframe['forward_RE'] > dataframe['prices_SB'], 0,
                                np.where(dataframe['forward_RE'] < dataframe['prices_SB'], 2, 1))

    epsilon = 0.1
    dataframe['price_diff'] = dataframe['forward_RE'] - dataframe['prices_SB']
    dataframe['balancing_state'] = balancing_states
    upwards_diff = dataframe['price_diff'][:HOURS_PER_YEAR][dataframe['balancing_state'] == 2].apply(
        lambda x: np.log(-x + epsilon))
    downwards_diff = dataframe['price_diff'][:HOURS_PER_YEAR][dataframe['balancing_state'] == 0].apply(
        lambda x: np.log(x + epsilon))

    # Find the best ARMA parameters for each state
    arma_model_up = auto_arima(upwards_diff[-HOURS_PER_MONTH:])

    arma_model_dw = auto_arima(downwards_diff[-HOURS_PER_MONTH:])

    print("Initial model orders: ", arma_model_up.order, arma_model_dw.order)

    hapd_model = HAPDModel.load('hapd_hmin50')
    results_hapd = hapd_model.load_results(flag=ORIGINAL)

    transition_matrices = build_hour_specific_transition_matrices(dataframe['balancing_state'][:HOURS_PER_YEAR].values,
                                                                  model_order=1)

    generate_scenarios_whole_year_multiprocessing_2(hapd_model, transition_matrices, copy.deepcopy(arma_model_up),
                                                    copy.deepcopy(arma_model_dw),
                                                    epsilon,
                                                    dataframe['price_diff'].values, balancing_states, num_scenarios=100)

    print("Total execution time: ", time.time() - initial_time)
