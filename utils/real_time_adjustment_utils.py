from utils.constants import *
from gurobipy import quicksum


def compute_objective_variable_bids(t, hours_left, p_adj, up, dw, forward_bids, prices_F, prices_S, prices_B):
    return quicksum(forward_bids[i] * prices_F[t + i]
                    + PRICE_H * p_adj[i]
                    + dw[i] * prices_S[t + i]
                    - up[i] * prices_B[t + i]
                    for i in range(hours_left))


def compute_objective_fixed_bids(t, idx_start, hours_left, p_adj, up, dw, forward_bids, prices_F, prices_S, prices_B):
    return quicksum(forward_bids[t - idx_start + i] * prices_F[t + i]
                    + PRICE_H * p_adj[i]
                    + dw[i] * prices_S[t + i]
                    - up[i] * prices_B[t + i]
                    for i in range(hours_left))


def compute_objective_single_price_variable_bids(t, hours_left, p_adj, settlement, forward_bids, prices_F,
                                                 balancing_prices):
    return quicksum(forward_bids[i] * prices_F[t + i]
                    + PRICE_H * p_adj[i]
                    + settlement[i] * balancing_prices[t + i]
                    for i in range(hours_left))


def compute_objective_single_price_fixed_bids(t, idx_start, hours_left, p_adj, settlement, forward_bids, prices_F,
                                              balancing_prices):
    return quicksum(forward_bids[t - idx_start + i] * prices_F[t + i]
                    + PRICE_H * p_adj[i]
                    + settlement[i] * balancing_prices[t + i]
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
