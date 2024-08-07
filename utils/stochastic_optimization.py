import copy

import numpy as np
from matplotlib import pyplot as plt

from utils.DataLoader import DataLoader
from utils.constants import HOURS_PER_DAY


def build_hour_specific_transition_matrices(states, model_order=1):
    # Initialize a dictionary to store the count of transitions for each hour based on the model order
    transition_matrices = {hour: np.zeros((3,) * (model_order + 1), dtype=float) for hour in range(24)}
    # Count transitions based on the hour
    for i in range(model_order, len(states)):
        previous_states = tuple(states[i - model_order:i])
        current_state = states[i]
        transition_matrices[(i - 1) % 24][previous_states + (current_state,)] += 1
    # Normalize the count of transitions for each hour
    for hour, matrix in transition_matrices.items():
        # Iterate on all the possible previous states
        for previous_states in np.ndindex((3,) * model_order):
            # Normalize the count of transitions for each state
            transition_matrices[hour][previous_states] = (
                    transition_matrices[hour][previous_states] / transition_matrices[hour][
                previous_states].sum()) if transition_matrices[hour][previous_states].sum() > 0 else np.ones(3) / 3
    return transition_matrices


def generate_states(num_steps, initial_states, initial_hour, transition_matrices, model_order):
    if len(initial_states) < model_order:
        raise ValueError("Initial states length must be at least equal to the model_order")
    states = np.zeros(num_steps + model_order, dtype=int)
    states[:model_order] = initial_states
    hour = initial_hour

    for i in range(model_order, num_steps + model_order):
        previous_states = tuple(states[i - model_order:i])
        transition_probs = transition_matrices[hour][previous_states]
        states[i] = np.random.choice(3, p=transition_probs)
        hour = (hour + 1) % 24

    return states[model_order:]


def generate_scenario(num_steps, initial_state, initial_hour, transition_matrices, arma_model_up, arma_model_dw, epsilon, prices_diff, balancing_states):
    states = np.zeros(num_steps + 1, dtype=int)
    states[0] = initial_state
    hour = initial_hour
    prices = np.zeros(num_steps)
    for i in range(num_steps):
        state = np.random.choice(3, p=transition_matrices[hour][states[i]])
        states[i + 1] = state
        if state == 0:
            prices[i] = np.exp(arma_model_dw.predict(1)) - epsilon
        elif state == 2:
            prices[i] = -np.exp(arma_model_up.predict(1)) + epsilon
        if balancing_states[i] == 0:
            arma_model_dw.update(np.log(prices_diff[i] + epsilon))
        elif balancing_states[i] == 2:
            arma_model_up.update(np.log(-prices_diff[i] + epsilon))
        hour = (hour + 1) % 24
    return prices


def plot_scenarios(scenarios_file, nominal_wind):
    data_loader = DataLoader()
    prices_SB, prices_F, prices_forecast, features, features_red, realized, forecasted_prod = data_loader.build_all(
            nominal_wind)
    scenarios = np.load(scenarios_file)

    for hour in range(6):
        plt.figure(figsize=(10, 6))
        for i in range(20):
            plt.scatter(range(HOURS_PER_DAY), scenarios[hour][i], edgecolors='w', s=50, alpha=0.5)
            plt.plot(scenarios[hour][i], label='Scenario {}'.format(i + 1), linestyle='--', alpha=0.5)
        plt.plot(prices_SB[hour:HOURS_PER_DAY], label='Balancing prices', color='red')
        plt.plot(prices_F[hour:HOURS_PER_DAY], label='Forward prices', color='red', linestyle='-.')
        plt.xlabel('Hour', fontsize=12)
        plt.ylabel('Price', fontsize=12)
        plt.title('Markov models/ARIMA scenarios', fontsize=14, pad=20)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.legend(fontsize=10, loc='upper left')
        plt.savefig(f'../plots/balancing_prices_forecast/scenario_examples_hour{hour}.png')
        plt.show()
