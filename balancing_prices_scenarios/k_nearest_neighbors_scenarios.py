import time

import numpy as np
import pandas as pd

from models.HAPDModel import HAPDModel
from utils.constants import HOURS_PER_DAY, HOURS_PER_WEEK, HOURS_PER_YEAR


def hamming_distance(state_sequence1, state_sequence2, prices_sequence1, prices_sequence2):
    return np.sum(state_sequence1 != state_sequence2)


def euclidean_distance(state_sequence1, state_sequence2, prices_sequence1, prices_sequence2):
    return np.sqrt(np.sum((prices_sequence1 - prices_sequence2) ** 2))


def k_nearest_sequences(balancing_states_set, day_states, balancing_prices_set, day_prices, sequence_length, k, distance_metric=hamming_distance):
    # Compute the distance between the day_states and all the sequences in the balancing_states_set
    distances = np.array([distance_metric(day_states[-sequence_length:], balancing_states_set[i:i + sequence_length], day_prices[-sequence_length:], balancing_prices_set[i:i + sequence_length]) for i in
                          range(len(balancing_states_set) - sequence_length - HOURS_PER_DAY)])
    # Return the indexes of the k most similar sequences
    return np.argsort(distances)[:k]


def generate_scenarios(model, generation_length, k, sequence_length, balancing_states_set, prices_diff_set,
                       distance_metric=hamming_distance):
    scenarios = np.zeros((generation_length, k, HOURS_PER_DAY), dtype=int)
    idx_start = model.test_start_index
    idx_end = idx_start + HOURS_PER_WEEK

    for i in range(idx_start, idx_end):
        scenarios[i - idx_start] = [
            model.single_balancing_prices[j + sequence_length:j + sequence_length + HOURS_PER_DAY] for j
            in
            k_nearest_sequences(balancing_states_set[:i],
                                balancing_states_set[i - HOURS_PER_DAY:i],
                                prices_diff_set[:i],
                                prices_diff_set[i - HOURS_PER_DAY:i],
                                sequence_length, k, distance_metric)]
    np.save(f'../results/stochastic_optimization/scenarios_hmin{model.h_min}_knn_k10_slen{sequence_length}.npy', scenarios)


if __name__ == '__main__':
    initial_time = time.time()

    dataframe = pd.read_csv(
        "../data/2022_2023/2022_2023_data.csv")

    balancing_states = np.where(dataframe['forward_RE'] > dataframe['prices_SB'], 0,
                                np.where(dataframe['forward_RE'] < dataframe['prices_SB'], 2, 1))

    epsilon = 0.1

    dataframe['balancing_state'] = balancing_states
    dataframe['price_diff'] = dataframe['forward_RE'] - dataframe['prices_SB']
    upwards_diff = dataframe['price_diff'][:HOURS_PER_YEAR][dataframe['balancing_state'] == 2].apply(
        lambda x: np.log(-x + epsilon))
    downwards_diff = dataframe['price_diff'][:HOURS_PER_YEAR][dataframe['balancing_state'] == 0].apply(
        lambda x: np.log(x + epsilon))

    hapd_model = HAPDModel.load('HAPD_model_22_23_hmin50_day')
    results_hapd = hapd_model.evaluate(HOURS_PER_WEEK)
    # best_adj_hapd = hapd_model.best_adjustment(results_hapd)
    # rule_based_adj_hapd = hapd_model.rule_based_adjustment(results_hapd)
    # mpc_adj_hapd = hapd_model.MPC_adjustment_with_naive_balancing_forecasts(results_hapd, 24,
    #                                                                         info_on_current_hour=False)

    for sequence_length in [2, 4, 6, 8, 12, 24]:
        generate_scenarios(hapd_model, HOURS_PER_WEEK, 10, sequence_length, balancing_states, dataframe['price_diff'], distance_metric=hamming_distance)
    print("Total execution time: ", time.time() - initial_time)