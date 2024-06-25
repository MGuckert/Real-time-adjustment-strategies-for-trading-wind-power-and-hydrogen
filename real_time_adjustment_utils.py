from constants import *
import gurobipy as gp
from gurobipy import GRB, quicksum


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
