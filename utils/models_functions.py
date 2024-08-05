import numpy as np
from matplotlib import pyplot as plt

from models.DeterministicModel import DeterministicModel
from models.DeterministicModelBalancing import DeterministicModelBalancing
from models.HAPDModel import HAPDModel
from models.HindsightModel import HindsightModel
from utils.constants import HOURS_PER_YEAR, ORIGINAL, HOURS_PER_MONTH
from utils.Result import Result


def generate_all_results_22_23_models():
    for h_min in [50, 100, 150, 200]:
        deterministic_model = DeterministicModel('Deterministic_model_22_23_hmin{}'.format(h_min), HOURS_PER_YEAR,
                                                 datafile='../data/2022_2023/2022_2023_data.csv', h_min=h_min)
        results = deterministic_model.evaluate(HOURS_PER_YEAR)
        best_adj = deterministic_model.best_adjustment(results)
        rule_based_adj = deterministic_model.rule_based_adjustment(results)
        mpc_results = deterministic_model.MPC_adjustment(results)
        deterministic_model.save_results(results, 'original')
        deterministic_model.save_results(mpc_results, 'mpc_adj')
        deterministic_model.save_results(best_adj, 'best_adj')
        deterministic_model.save_results(rule_based_adj, 'rule_based_adj')

        print('Deterministic model with h_min = {} done'.format(h_min))

        hindsight_model = HindsightModel('Hindsight_model_22_23_hmin{}'.format(h_min), HOURS_PER_YEAR,
                                         datafile='../data/2022_2023/2022_2023_data.csv', h_min=h_min)
        hindsight_results = hindsight_model.evaluate(HOURS_PER_YEAR)
        hindsight_best_adj = hindsight_model.best_adjustment(hindsight_results)
        hindsight_rule_based_adj = hindsight_model.rule_based_adjustment(hindsight_results)
        hindsight_mpc_results = hindsight_model.MPC_adjustment(hindsight_results)
        hindsight_model.save_results(hindsight_results, 'original')
        hindsight_model.save_results(hindsight_mpc_results, 'mpc_adj')
        hindsight_model.save_results(hindsight_best_adj, 'best_adj')
        hindsight_model.save_results(hindsight_rule_based_adj, 'rule_based_adj')

        print('Hindsight model with h_min = {} done'.format(h_min))

        hapd_model = HAPDModel('HAPD_model_22_23_hmin{}'.format(h_min), HOURS_PER_YEAR, h_min=h_min,
                               datafile='../data/2022_2023/2022_2023_data.csv')
        hapd_model.train()
        hapd_results = hapd_model.evaluate(HOURS_PER_YEAR)
        hapd_best_adj = hapd_model.best_adjustment(hapd_results)
        hapd_rule_based_adj = hapd_model.rule_based_adjustment(hapd_results)
        hapd_mpc_results = hapd_model.MPC_adjustment(hapd_results)
        hapd_model.save_results(hapd_results, 'original')
        hapd_model.save_results(hapd_mpc_results, 'mpc_adj')
        hapd_model.save_results(hapd_best_adj, 'best_adj')
        hapd_model.save_results(hapd_rule_based_adj, 'rule_based_adj')

        print('HAPD model with h_min = {} done'.format(h_min))


def compute_and_plot_det_with_naive_balancing_22_23(lags):
    for h_min in [50, 100, 150, 200]:

        objectives = []

        default_deterministic_model = DeterministicModel.load('Deterministic_model_22_23_hmin{}'.format(h_min),
                                                              DeterministicModel)
        results = default_deterministic_model.load_results(ORIGINAL)
        default_objective = results.get_total_objective()

        for lag in lags:
            deterministic_model = DeterministicModelBalancing('Deterministic_model_22_23_balancing_lag_{}'.format(lag),
                                                              HOURS_PER_YEAR, lag,
                                                              datafile='../data/2022_2023/2022_2023_data.csv')
            results = deterministic_model.evaluate(HOURS_PER_YEAR)
            objectives.append(results.get_total_objective())

        plt.figure(figsize=(10, 6))
        plt.scatter(lags[1:], objectives[1:], color='darkorange', edgecolors='w', s=100, alpha=0.75,
                    label='Deterministic model with naive balancing prices forecast')
        plt.plot(lags[1:], objectives[1:], linestyle='-', color='darkorange', alpha=0.5)
        plt.axhline(y=objectives[0], color='darkgreen', linestyle='-.', label='Hindsight')
        plt.axhline(y=default_objective, color='red', linestyle='-.',
                    label='Deterministic model without balancing prices')
        plt.xlabel('Lag', fontsize=14)
        plt.ylabel('Objective', fontsize=14)
        plt.title(f'Deterministic model objective for naive balancing prices forecast (2022-23, Hmin = {h_min})',
                  fontsize=16, pad=20)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        # Add a dashed line at the objective of deterministic model without balancing prices

        plt.tight_layout()
        plt.legend(fontsize=12, loc='upper right', bbox_to_anchor=(1, 0.95))
        plt.savefig(f'../plots/2022_2023/deterministic_model_22_23_naive_forecasts_hmin{h_min}.png')
        plt.show()


def plot_rmse_balancing_shifted_22_23(balancing_prices, max_lag=HOURS_PER_MONTH):
    lags = [i for i in range(1, max_lag)]
    list_rmse = []
    list_mae = []
    for lag in lags:
        balancing_prices_shifted = balancing_prices[lag:]
        mae = np.mean(np.abs(balancing_prices_shifted - balancing_prices[:len(balancing_prices) - lag]))
        rmse = np.sqrt(np.mean((balancing_prices_shifted - balancing_prices[:len(balancing_prices) - lag]) ** 2))
        list_rmse.append(rmse)
        list_mae.append(mae)
        
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(lags, list_rmse, color='darkorange', edgecolors='w', s=10, alpha=0.75,
                label='Deterministic model with naive balancing prices forecast')
    plt.plot(lags, list_rmse, linestyle='-', color='darkorange', alpha=0.5)
    plt.xlabel('Lag', fontsize=14)
    plt.ylabel('RMSE', fontsize=14)
    plt.title('RMSE between balancing prices and lagged balancing prices (2022-23)', fontsize=16, pad=20)
    plt.xticks([24 * i for i in range(8)], fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('../plots/balancing_prices_forecast/rmse_lagged_balancing_prices_week_22_23.png')
    plt.show()
