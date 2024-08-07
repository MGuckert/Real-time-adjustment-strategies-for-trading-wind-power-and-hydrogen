from utils.constants import *
from gurobipy import quicksum
import gurobipy as gp


def compute_objective_variable_bids(t, hours_left, p_adj, settlement, forward_bids, prices_F,
                                    balancing_prices):
    return quicksum(forward_bids[i] * prices_F[t + i]
                    + PRICE_H * p_adj[i]
                    + settlement[i] * balancing_prices[t + i]
                    for i in range(hours_left))


def compute_objective_fixed_bids(t, idx_start, hours_left, p_adj, settlement, forward_bids, prices_F,
                                 balancing_prices):
    return quicksum(forward_bids[t - idx_start + i] * prices_F[t + i]
                    + PRICE_H * p_adj[i]
                    + settlement[i] * balancing_prices[t + i]
                    for i in range(hours_left))


def compute_objective_fixed_bids_naive_balancing_prices(t, idx_start, hours_left, p_adj, settlement, forward_bids,
                                                        prices_F,
                                                        balancing_prices, lag, info_on_current_hour=True):
    if info_on_current_hour:
        return quicksum(forward_bids[t - idx_start + i] * prices_F[t + i]
                        + PRICE_H * p_adj[i]
                        + settlement[i] * balancing_prices[(t + i) if i == 0 else t + i - lag]
                        for i in range(hours_left))
    else:
        return quicksum(forward_bids[t - idx_start + i] * prices_F[t + i]
                        + PRICE_H * p_adj[i]
                        + settlement[i] * balancing_prices[(t + i - 1) if i == 0 else t + i - lag]
                        for i in range(hours_left))


def compute_objective_fixed_bids_balancing_prices_forecasts_scenarios(hours_left, p_adj, settlements,
                                                                     balancing_prices_scenarios):
    objective = gp.LinExpr()
    objective.addTerms([PRICE_H] * hours_left, p_adj)
    for i in range(len(balancing_prices_scenarios)):
        objective.addTerms(balancing_prices_scenarios[i][:hours_left] / len(balancing_prices_scenarios), settlements)
    return objective


def compute_objective_fixed_bids_balancing_prices_forecasts_single_scenario(hours_left, p_adj, settlements,
                                                                            balancing_prices_scenario):
    objective = gp.LinExpr()
    objective.addTerms([PRICE_H] * hours_left, p_adj)
    objective.addTerms(balancing_prices_scenario[:hours_left], settlements)
    return objective


def compute_objective_fixed_bids_balancing_prices_no_info_current_hour_threshold(t, idx_start, hours_left, p_adj,
                                                                                 settlement,
                                                                                 forward_bids, prices_F,
                                                                                 balancing_prices, lag, threshold):
    return quicksum(forward_bids[t - idx_start + i] * prices_F[t + i]
                    + PRICE_H * p_adj[i]
                    + settlement[i] * balancing_prices[(t - 1) if i < threshold else t + i - lag]
                    for i in range(hours_left))


def get_hydro_opt(single_balancing_price, p_h_max):
    """
    Get optimal adjustment at given balancing prices.

    Args:
        p_h_max: Maximum hydrogen production.
        single_balancing_price: Single balancing price.

    Returns:
        float: Optimal adjustment value.
    """
    if single_balancing_price < PRICE_H:
        return p_h_max
    else:
        return 0
