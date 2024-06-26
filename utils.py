import numpy as np


def get_total_objective(result):
    return np.sum(result["objectives"])