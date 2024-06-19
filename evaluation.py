import numpy as np
import pandas as pd

from constants import *

def get_forward_1(qF, features, t, fm):
    """
    Get forward market bid for model with 1 parameter per feature (General architecture).

    Args:
        qF (DataFrame): Quantities for forward market.
        features (DataFrame): Feature data.
        t (int): Time index.
        fm (bool): Flag for model.

    Returns:
        float: Forward market bid.
    """
    if fm:
        return np.sum([qF.iloc[0, i] * features[i][t] for i in range(len(features))]) + qF.iloc[0, -1]
    else:
        return np.sum([qF.iloc[0, i] * features.iloc[t, i] for i in range(len(features.columns))]) + qF.iloc[0, -1]


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


def get_scaled_objs(list_of_results):
    """
    Get total revenue scaled to millions.

    Args:
        list_of_results (list): List of result dictionaries.

    Returns:
        list: Total revenue scaled to millions.
    """
    return [np.sum(result['obj']) / 1e6 for result in list_of_results]


def get_remaining_planned_hydro(df, features, max_elec, i):
    """
    Get the remaining amount of hydrogen scheduled in a given hour.

    Args:
        df (DataFrame): Data frame containing hydrogen schedules.
        features (DataFrame): Feature data.
        max_elec (float): Maximum electricity.
        i (int): Time index.

    Returns:
        float: Remaining amount of hydrogen scheduled.
    """
    hour_of_day = (i % 24)
    if hour_of_day == 23:
        return 0
    remaining_hours = 23 - hour_of_day
    return np.sum([get_hydro_24(df, features, max_elec, i + j) for j in range(1, remaining_hours)])


def test_initial_plan(df_f, df_h, idx_start, idx_end, general=False, fm=False, reduced_features=False):
    """
    Perform complete evaluation for general and hourly models in a given time period.

    Args:
        df_f (DataFrame): Forward market quantities.
        df_h (DataFrame): Hydrogen quantities.
        idx_start (int): Start index.
        idx_end (int): End index.
        general (bool): Flag for general model.
        fm (bool): Flag for model.
        reduced_features (bool): Flag for reduced features.

    Returns:
        dict: Results of the evaluation.
    """
    (prices_B, prices_S, prices_F, prices_forecast, features, features_red,
     realized, price_H, max_elec, nominal_wind, penalty, min_production,
     forecast_production) = import_consts()

    forward_bids = []
    ds = []
    h_prods = []
    ups = []
    dws = []
    objs = []
    missing_productions = []
    missing_production = 0
    daily_count = 0

    for i in range(idx_start, idx_end):
        hour_of_day = (i % 24)
        if hour_of_day == 0 and i != idx_start:
            missing_production = np.maximum(min_production - daily_count, 0)
            daily_count = 0

        if general:
            if fm:
                forward_bid = get_forward_1(df_f, [forecast_production, prices_F], i, fm)
                h_prod = get_hydro_1(df_h, [forecast_production, prices_F], max_elec, i, fm)
            else:
                if reduced_features:
                    forward_bid = get_forward_1(df_f, features_red, i, fm)
                    h_prod = get_hydro_1(df_h, features_red, max_elec, i, fm)
                else:
                    forward_bid = get_forward_1(df_f, features, i, fm)
                    h_prod = get_hydro_1(df_h, features, max_elec, i, fm)
        else:
            if fm:
                forward_bid = get_forward_24(df_f, [forecast_production, prices_F], i, fm)
                h_prod = get_hydro_24(df_h, [forecast_production, prices_F], max_elec, i, fm)
            else:
                if reduced_features:
                    forward_bid = get_forward_24(df_f, features_red, i, fm)
                    h_prod = get_hydro_24(df_h, features_red, max_elec, i, fm)
                else:
                    forward_bid = get_forward_24(df_f, features, i, fm)
                    h_prod = get_hydro_24(df_h, features, max_elec, i, fm)

        forward_bid = np.minimum(forward_bid, nominal_wind)
        forward_bid = np.maximum(forward_bid, -max_elec)
        d = realized[i] - forward_bid

        h_prod = np.maximum(h_prod, 0)
        h_prod = np.minimum(max_elec, h_prod)
        daily_count += h_prod

        settlementd = realized[i] - forward_bid - h_prod

        up = np.maximum(-settlementd, 0)
        dw = np.maximum(settlementd, 0)
        obj = (
                forward_bid * prices_F[i]
                + price_H * h_prod
                + dw * prices_S[i]
                - up * prices_B[i]
                - missing_production * penalty
        )

        forward_bids.append(forward_bid)
        ds.append(d)
        h_prods.append(h_prod)
        ups.append(up)
        dws.append(dw)
        missing_productions.append(missing_production)
        missing_production = 0
        objs.append(obj)

    results = {
        "forward_bid": forward_bids,
        "d": ds,
        "h_prod": h_prods,
        "up": ups,
        "dw": dws,
        "missing_production": missing_productions,
        "obj": objs,
    }
    return results


def test_initial_plan_changing_qs(init_filename, cqs_filename, idx_start, idx_end, general=False, fm=False,
                                  reduced_features=False, weekly=False):
    """
    Perform complete evaluation for general and hourly models with retraining in a given time period.

    Args:
        init_filename (str): Initial filename for retrained data.
        cqs_filename (str): Filename pattern for retrained data.
        idx_start (int): Start index.
        idx_end (int): End index.
        general (bool): Flag for general model.
        fm (bool): Flag for model.
        reduced_features (bool): Flag for reduced features.
        weekly (bool): Flag for weekly retraining.

    Returns:
        dict: Results of the evaluation.
    """
    (prices_B, prices_S, prices_F, prices_forecast, features, features_red,
     realized, price_H, max_elec, nominal_wind, penalty, min_production,
     forecast_production) = import_consts()

    forward_bids = []
    ds = []
    h_prods = []
    ups = []
    dws = []
    objs = []
    missing_productions = []
    missing_production = 0
    daily_count = 0

    df_retrained = pd.read_csv(init_filename)

    for i in range(idx_start, idx_end):
        hour_of_day = (i % 24)
        if hour_of_day == 0 and i != idx_start:
            missing_production = np.maximum(0, min_production - daily_count)
            daily_count = 0

        j = i - idx_start
        if weekly:
            if j != 0 and j % 168 == 0:
                df_retrained = pd.read_csv(f'{cqs_filename}_we{round(j / 168, 0):.0f}.csv')
        else:
            if j != 0 and j % 720 == 0:
                df_retrained = pd.read_csv(f'{cqs_filename}_mo{round(j / 720, 0):.0f}.csv')

        if fm:
            df_f = df_retrained[[f"qF{i + 1}" for i in range(3)]]
            df_h = df_retrained[[f"qH{i + 1}" for i in range(3)]]
        else:
            if reduced_features:
                df_f = df_retrained[[f"qF{i + 1}" for i in range(len(features_red.columns) + 1)]]
                df_h = df_retrained[[f"qH{i + 1}" for i in range(len(features_red.columns) + 1)]]
            else:
                df_f = df_retrained[[f"qF{i + 1}" for i in range(len(features.columns) + 1)]]
                df_h = df_retrained[[f"qH{i + 1}" for i in range(len(features.columns) + 1)]]

        if general:
            if fm:
                forward_bid = get_forward_1(df_f, [forecast_production, prices_F], i, fm)
                h_prod = get_hydro_1(df_h, [forecast_production, prices_F], max_elec, i, fm)
            else:
                if reduced_features:
                    forward_bid = get_forward_1(df_f, features_red, i, fm)
                    h_prod = get_hydro_1(df_h, features_red, max_elec, i, fm)
                else:
                    forward_bid = get_forward_1(df_f, features, i, fm)
                    h_prod = get_hydro_1(df_h, features, max_elec, i, fm)
        else:
            if fm:
                forward_bid = get_forward_24(df_f, [forecast_production, prices_F], i, fm)
                h_prod = get_hydro_24(df_h, [forecast_production, prices_F], max_elec, i, fm)
            else:
                if reduced_features:
                    forward_bid = get_forward_24(df_f, features_red, i, fm)
                    h_prod = get_hydro_24(df_h, features_red, max_elec, i, fm)
                else:
                    forward_bid = get_forward_24(df_f, features, i, fm)
                    h_prod = get_hydro_24(df_h, features, max_elec, i, fm)

        d = realized[i] - forward_bid

        forward_bid = np.minimum(forward_bid, max_elec)
        forward_bid = np.maximum(forward_bid, -max_elec)
        h_prod = np.maximum(h_prod, 0)
        h_prod = np.minimum(max_elec, h_prod)
        daily_count += h_prod
        settlementd = realized[i] - forward_bid - h_prod

        up = np.maximum(-settlementd, 0)
        dw = np.maximum(settlementd, 0)
        obj = (
                forward_bid * prices_F[i]
                + price_H * h_prod
                + dw * prices_S[i]
                - up * prices_B[i]
                - missing_production * penalty
        )

        forward_bids.append(forward_bid)
        ds.append(d)
        h_prods.append(h_prod)
        ups.append(up)
        dws.append(dw)
        missing_productions.append(missing_production)
        missing_production = 0
        objs.append(obj)

    results = {
        "forward_bid": forward_bids,
        "d": ds,
        "h_prod": h_prods,
        "up": ups,
        "dw": dws,
        "missing_production": missing_productions,
        "obj": objs,
    }
    return results


def test_fixed(forward, hydrogen, idx_start, idx_end, h_min):
    """
    Perform complete evaluation for deterministic with all bids accepted or hindsight models.

    Args:
        forward (list): Forward bids.
        hydrogen (list): Hydrogen schedules.
        idx_start (int): Start index.
        idx_end (int): End index.

    Returns:
        dict: Results of the evaluation.
    """
    (prices_B, prices_S, prices_F, prices_forecast, features, features_red,
     realized, price_H, max_elec, nominal_wind, penalty, min_production,
     forecast_production) = import_consts()

    min_production = h_min

    forward_bids = []
    ds = []
    h_prods = []
    ups = []
    dws = []
    objs = []
    missing_productions = []
    missing_production = 0
    daily_count = 0

    for i, t in enumerate(range(idx_start, idx_end)):
        hour_of_day = (t % 24)
        if hour_of_day == 0 and t != idx_start:
            missing_production = np.maximum(min_production - daily_count, 0)
            daily_count = 0

        forward_bid = forward[i]
        d = realized[t] - forward_bid
        h_prod = hydrogen[i]

        settlementd = realized[t] - forward_bid - h_prod
        daily_count += h_prod

        up = np.maximum(-settlementd, 0)
        dw = np.maximum(settlementd, 0)
        obj = (
                forward_bid * prices_F[t]
                + price_H * h_prod
                + dw * prices_S[t]
                - up * prices_B[t]
                - missing_production * penalty
        )

        forward_bids.append(forward_bid)
        ds.append(d)
        h_prods.append(h_prod)
        ups.append(up)
        missing_productions.append(missing_production)
        missing_production = 0
        dws.append(dw)
        objs.append(obj)

    results = {
        "forward_bid": forward_bids,
        "d": ds,
        "h_prod": h_prods,
        "up": ups,
        "dw": dws,
        "missing_production": missing_productions,
        "obj": objs,
    }

    return results


def test_det(forward, hydrogen, idx_start, idx_end):
    """
    Perform complete evaluation for deterministic model with 5€/MW buffer in forward bids.

    Args:
        forward (list): Forward bids.
        hydrogen (list): Hydrogen schedules.
        idx_start (int): Start index.
        idx_end (int): End index.

    Returns:
        dict: Results of the evaluation.
    """
    (prices_B, prices_S, prices_F, prices_forecast, features, features_red,
     realized, price_H, max_elec, nominal_wind, penalty, min_production,
     forecast_production) = import_consts()

    forward_bids = []
    ds = []
    h_prods = []
    ups = []
    dws = []
    objs = []
    missing_productions = []
    missing_production = 0
    daily_count = 0

    for i, t in enumerate(range(idx_start, idx_end)):
        hour_of_day = (t % 24)
        if hour_of_day == 0 and t != idx_start:
            missing_production = np.maximum(min_production - daily_count, 0)
            daily_count = 0

        if forward[i] >= 0:
            if prices_F[t] > (prices_forecast[t] - 5):
                forward_bid = forward[i]
            else:
                forward_bid = 0
        else:
            if prices_F[t] < (prices_forecast[t] + 5):
                forward_bid = forward[i]
            else:
                forward_bid = 0

        d = realized[t] - forward_bid
        h_prod = hydrogen[i]

        settlementd = realized[t] - forward_bid - h_prod
        daily_count += h_prod

        up = np.maximum(-settlementd, 0)
        dw = np.maximum(settlementd, 0)
        obj = (
                forward_bid * prices_F[t]
                + price_H * h_prod
                + dw * prices_S[t]
                - up * prices_B[t]
                - missing_production * penalty
        )

        forward_bids.append(forward_bid)
        ds.append(d)
        h_prods.append(h_prod)
        ups.append(up)
        missing_productions.append(missing_production)
        missing_production = 0
        dws.append(dw)
        objs.append(obj)

    results = {
        "forward_bid": forward_bids,
        "d": ds,
        "h_prod": h_prods,
        "up": ups,
        "dw": dws,
        "missing_production": missing_productions,
        "obj": objs,
    }

    return results


def test_price_domain(df, idx_start, idx_end, h_min, general=False, fm=False, reduced_features=False):
    """
    Perform complete evaluation for general and hourly models with price domains in a given time period.

    Args:
        df (DataFrame): Data frame containing quantities.
        idx_start (int): Start index.
        idx_end (int): End index.
        general (bool): Flag for general model.
        fm (bool): Flag for model.
        reduced_features (bool): Flag for reduced features.

    Returns:
        dict: Results of the evaluation.
    """
    (prices_B, prices_S, prices_F, prices_forecast, features, features_red,
     realized, price_H, max_elec, nominal_wind, penalty, min_production,
     forecast_production) = import_consts()

    min_production = h_min

    forward_bids = []
    ds = []
    h_prods = []
    ups = []
    dws = []
    objs = []
    missing_productions = []
    missing_production = 0
    daily_count = 0

    for i in range(idx_start, idx_end):
        hour_of_day = (i % 24)
        if hour_of_day == 0 and i != idx_start:
            missing_production = np.maximum(min_production - daily_count, 0)
            daily_count = 0

        if prices_F[i] < price_H:
            domain = 1
        elif prices_F[i] < TOP_DOMAIN:
            domain = 2
        else:
            domain = 3

        if fm:
            df_f = df[[f"qF{domain}_{i + 1}" for i in range(3)]]
            df_h = df[[f"qH{domain}_{i + 1}" for i in range(3)]]
        else:
            if reduced_features:
                df_f = df[[f"qF{domain}_{i + 1}" for i in range(len(features_red.columns) + 1)]]
                df_h = df[[f"qH{domain}_{i + 1}" for i in range(len(features_red.columns) + 1)]]
            else:
                df_f = df[[f"qF{domain}_{i + 1}" for i in range(len(features.columns) + 1)]]
                df_h = df[[f"qH{domain}_{i + 1}" for i in range(len(features.columns) + 1)]]

        if general:
            if fm:
                forward_bid = get_forward_1(df_f, [forecast_production, prices_F], i, fm)
                h_prod = get_hydro_1(df_h, [forecast_production, prices_F], max_elec, i, fm)
            else:
                if reduced_features:
                    forward_bid = get_forward_1(df_f, features_red, i, fm)
                    h_prod = get_hydro_1(df_h, features_red, max_elec, i, fm)
                else:
                    forward_bid = get_forward_1(df_f, features, i, fm)
                    h_prod = get_hydro_1(df_h, features, max_elec, i, fm)
        else:
            if fm:
                forward_bid = get_forward_24(df_f, [forecast_production, prices_F], i, fm)
                h_prod = get_hydro_24(df_h, [forecast_production, prices_F], max_elec, i, fm)
            else:
                if reduced_features:
                    forward_bid = get_forward_24(df_f, features_red, i, fm)
                    h_prod = get_hydro_24(df_h, features_red, max_elec, i, fm)
                else:
                    forward_bid = get_forward_24(df_f, features, i, fm)
                    h_prod = get_hydro_24(df_h, features, max_elec, i, fm)

        forward_bid = np.minimum(forward_bid, max_elec)
        forward_bid = np.maximum(forward_bid, -max_elec)
        d = realized[i] - forward_bid

        h_prod = np.maximum(h_prod, 0)
        h_prod = np.minimum(max_elec, h_prod)
        daily_count += h_prod
        settlementd = realized[i] - forward_bid - h_prod
        up = np.maximum(-settlementd, 0)
        dw = np.maximum(settlementd, 0)
        obj = (
                forward_bid * prices_F[i]
                + price_H * h_prod
                + dw * prices_S[i]
                - up * prices_B[i]
                - missing_production * penalty
        )

        forward_bids.append(forward_bid)
        ds.append(d)
        h_prods.append(h_prod)
        ups.append(up)
        dws.append(dw)
        missing_productions.append(missing_production)
        missing_production = 0
        objs.append(obj)

    results = {
        "forward_bid": forward_bids,
        "d": ds,
        "h_prod": h_prods,
        "up": ups,
        "dw": dws,
        "missing_production": missing_productions,
        "obj": objs,
    }
    return results


def test_price_domain_changing_qs(init_filename, cqs_filename, idx_start, idx_end, general=False, fm=False,
                                  reduced_features=False, weekly=False):
    """
    Perform complete evaluation for general and hourly models with price domains and retraining in a given time period.

    Args:
        init_filename (str): Initial filename for retrained data.
        cqs_filename (str): Filename pattern for retrained data.
        idx_start (int): Start index.
        idx_end (int): End index.
        general (bool): Flag for general model.
        fm (bool): Flag for model.
        reduced_features (bool): Flag for reduced features.
        weekly (bool): Flag for weekly retraining.

    Returns:
        dict: Results of the evaluation.
    """
    (prices_B, prices_S, prices_F, prices_forecast, features, features_red,
     realized, price_H, max_elec, nominal_wind, penalty, min_production,
     forecast_production) = import_consts()

    forward_bids = []
    ds = []
    h_prods = []
    ups = []
    dws = []
    objs = []
    missing_productions = []
    missing_production = 0
    daily_count = 0

    df_retrained = pd.read_csv(init_filename)

    for i in range(idx_start, idx_end):
        hour_of_day = (i % 24)
        if hour_of_day == 0 and i != idx_start:
            missing_production = np.maximum(0, min_production - daily_count)
            daily_count = 0

        j = i - idx_start
        if weekly:
            if j != 0 and j % 168 == 0:
                df_retrained = pd.read_csv(f'{cqs_filename}_we{round(j / 168, 0):.0f}.csv')
        else:
            if j != 0 and j % 720 == 0:
                df_retrained = pd.read_csv(f'{cqs_filename}_mo{round(j / 720, 0):.0f}.csv')

        if prices_F[i] < price_H:
            domain = 1
        elif prices_F[i] < TOP_DOMAIN:
            domain = 2
        else:
            domain = 3

        if fm:
            df_f = df_retrained[[f"qF{domain}_{i + 1}" for i in range(3)]]
            df_h = df_retrained[[f"qH{domain}_{i + 1}" for i in range(3)]]
        else:
            if reduced_features:
                df_f = df_retrained[[f"qF{domain}_{i + 1}" for i in range(len(features_red.columns) + 1)]]
                df_h = df_retrained[[f"qH{domain}_{i + 1}" for i in range(len(features_red.columns) + 1)]]
            else:
                df_f = df_retrained[[f"qF{domain}_{i + 1}" for i in range(len(features.columns) + 1)]]
                df_h = df_retrained[[f"qH{domain}_{i + 1}" for i in range(len(features.columns) + 1)]]

        if general:
            if fm:
                forward_bid = get_forward_1(df_f, [forecast_production, prices_F], i, fm)
                h_prod = get_hydro_1(df_h, [forecast_production, prices_F], max_elec, i, fm)
            else:
                if reduced_features:
                    forward_bid = get_forward_1(df_f, features_red, i, fm)
                    h_prod = get_hydro_1(df_h, features_red, max_elec, i, fm)
                else:
                    forward_bid = get_forward_1(df_f, features, i, fm)
                    h_prod = get_hydro_1(df_h, features, max_elec, i, fm)
        else:
            if fm:
                forward_bid = get_forward_24(df_f, [forecast_production, prices_F], i, fm)
                h_prod = get_hydro_24(df_h, [forecast_production, prices_F], max_elec, i, fm)
            else:
                if reduced_features:
                    forward_bid = get_forward_24(df_f, features_red, i, fm)
                    h_prod = get_hydro_24(df_h, features_red, max_elec, i, fm)
                else:
                    forward_bid = get_forward_24(df_f, features, i, fm)
                    h_prod = get_hydro_24(df_h, features, max_elec, i, fm)

        d = realized[i] - forward_bid

        forward_bid = np.minimum(forward_bid, max_elec)
        forward_bid = np.maximum(forward_bid, -max_elec)
        h_prod = np.maximum(h_prod, 0)
        h_prod = np.minimum(max_elec, h_prod)
        daily_count += h_prod
        settlementd = realized[i] - forward_bid - h_prod

        up = np.maximum(-settlementd, 0)
        dw = np.maximum(settlementd, 0)
        obj = (
                forward_bid * prices_F[i]
                + price_H * h_prod
                + dw * prices_S[i]
                - up * prices_B[i]
                - missing_production * penalty
        )

        forward_bids.append(forward_bid)
        ds.append(d)
        h_prods.append(h_prod)
        ups.append(up)
        dws.append(dw)
        missing_productions.append(missing_production)
        missing_production = 0
        objs.append(obj)

    results = {
        "forward_bid": forward_bids,
        "d": ds,
        "h_prod": h_prods,
        "up": ups,
        "dw": dws,
        "missing_production": missing_productions,
        "obj": objs,
    }
    return results