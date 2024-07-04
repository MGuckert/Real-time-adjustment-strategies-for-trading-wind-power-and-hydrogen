import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_forward_24(qF, features, t, fm):
    """
    Get forward market bid for model with 24 parameters per feature (Hourly architecture).

    Args:
        qF (DataFrame): Quantities for forward market.
        features (DataFrame): Feature data.
        t (int): Time index.
        fm (bool): Flag for model.

    Returns:
        float: Forward market bid.
    """
    hour_of_day = (t % 24)
    if fm:
        return np.sum([qF.iloc[hour_of_day, i] * features[i][t] for i in range(len(features))]) + qF.iloc[
            hour_of_day, -1]
    else:
        return np.sum([qF.iloc[hour_of_day, i] * features.iloc[t, i] for i in range(len(features.columns))]) + qF.iloc[
            hour_of_day, -1]


def get_hydro_1(qH, features, max_elec, t, fm):
    """
    Get hydrogen schedule for model with 1 parameter per feature (General architecture).

    Args:
        qH (DataFrame): Quantities for hydrogen.
        features (DataFrame): Feature data.
        max_elec (float): Maximum electricity.
        t (int): Time index.
        fm (bool): Flag for model.

    Returns:
        float: Hydrogen schedule.
    """
    if fm:
        return np.minimum(max_elec, np.maximum(0,
                                               np.sum([qH.iloc[0, i] * features[i][t] for i in range(len(features))]) +
                                               qH.iloc[0, -1]))
    else:
        return np.minimum(max_elec, np.maximum(0, np.sum(
            [qH.iloc[0, i] * features.iloc[t, i] for i in range(len(features.columns))]) + qH.iloc[0, -1]))


def get_hydro_24(qH, features, max_elec, t, fm):
    """
    Get hydrogen schedule for model with 24 parameters per feature (Hourly architecture).

    Args:
        qH (DataFrame): Quantities for hydrogen.
        features (DataFrame): Feature data.
        max_elec (float): Maximum electricity.
        t (int): Time index.
        fm (bool): Flag for model.

    Returns:
        float: Hydrogen schedule.
    """
    hour_of_day = (t % 24)
    if fm:
        return np.minimum(max_elec, np.maximum(0, np.sum(
            [qH.iloc[hour_of_day, i] * features[i][t] for i in range(len(features))]) + qH.iloc[hour_of_day, -1]))
    else:
        return np.minimum(max_elec, np.maximum(0, np.sum(
            [qH.iloc[hour_of_day, i] * features.iloc[t, i] for i in range(len(features.columns))]) + qH.iloc[
                                                   hour_of_day, -1]))