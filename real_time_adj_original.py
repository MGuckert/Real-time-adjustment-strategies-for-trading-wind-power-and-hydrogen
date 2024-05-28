import copy
import numpy as np


def get_hydro_opt(d, lh, ls, lb, max_elec):
    """
    Get optimal adjustment at given balancing prices.

    Args:
        d (float): Some value (to be defined).
        lh (float): High price limit.
        ls (float): Low sell price.
        lb (float): Low buy price.
        max_elec (float): Maximum electricity.

    Returns:
        float: Optimal adjustment value.
    """
    if ls > lh:
        return 0
    elif lb < lh:
        return max_elec
    else:
        return np.minimum(max_elec, np.maximum(0, d))


def apply_upwards_adj(results_to_copy, idx_start, idx_end, printing=False):
    """
    Perform complete evaluation for upwards adjustment performed on a given model.

    Args:
        results_to_copy (dict): Results dictionary to copy.
        idx_start (int): Start index for evaluation.
        idx_end (int): End index for evaluation.
        printing (bool, optional): If True, prints adjustments. Defaults to False.

    Returns:
        dict: Results of the adjustment.
    """
    results = copy.deepcopy(results_to_copy)
    (prices_B, prices_S, prices_F, prices_forecast, features, features_red, realized, price_H, max_elec,
     nominal_wind, penalty, min_production, forecast_production) = import_consts()

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
        hour_of_day = (i % 24)
        if (hour_of_day == 0) and t != idx_start:
            missing_production = np.maximum(min_production - daily_count, 0)
            daily_count = 0

        forward_bid = results['forward_bid'][i]
        h_prod = results['h_prod'][i]

        d = realized[t] - forward_bid

        opt_h = get_hydro_opt(d, price_H, prices_S[t], prices_B[t], max_elec)

        if (opt_h > h_prod):
            if printing:
                print(f"i={i}, method 1: We changed from {h_prod} to {opt_h}")
            h_adj = opt_h
        else:
            h_adj = h_prod

        daily_count += h_adj
        settlementd = realized[t] - forward_bid - h_adj
        up = np.maximum(-settlementd, 0)
        dw = np.maximum(settlementd, 0)
        obj = (
                forward_bid * prices_F[t]
                + price_H * h_adj
                + dw * prices_S[t]
                - up * prices_B[t]
                - missing_production * penalty
        )

        forward_bids.append(forward_bid)
        ds.append(d)
        h_prods.append(h_adj)
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


def apply_up_and_dw_adj(results_to_copy, idx_start, idx_end, printing=False):
    """
    Perform complete evaluation for upwards and downwards adjustment performed on a given model.

    Args:
        results_to_copy (dict): Results dictionary to copy.
        idx_start (int): Start index for evaluation.
        idx_end (int): End index for evaluation.
        printing (bool, optional): If True, prints adjustments. Defaults to False.

    Returns:
        dict: Results of the adjustment.
    """
    results = copy.deepcopy(results_to_copy)
    (prices_B, prices_S, prices_F, prices_forecast, features, features_red, realized, price_H, max_elec,
     nominal_wind, penalty, min_production, forecast_production) = import_consts()

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
        hour_of_day = (i % 24)
        if (hour_of_day == 0) and t != idx_start:
            missing_production = np.maximum(min_production - daily_count, 0)
            daily_count = 0

        forward_bid = results['forward_bid'][i]
        h_prod = results['h_prod'][i]

        d = realized[t] - forward_bid

        opt_h = get_hydro_opt(d, price_H, prices_S[t], prices_B[t], max_elec)

        if (opt_h > h_prod):
            if printing:
                print(f"i={i}, method 1: We changed from {h_prod} to {opt_h}")
            h_adj = opt_h
        else:
            remaining_hours = 23 - hour_of_day
            if (hour_of_day == 23):
                remaining_planned = 0
            else:
                remaining_planned = np.sum([results['h_prod'][i + j] for j in range(remaining_hours + 1)])
            surplus = daily_count + remaining_planned - min_production
            wanted = h_prod - opt_h
            if surplus >= wanted:
                h_adj = opt_h
            else:
                h_adj = np.minimum(np.maximum(h_prod - surplus, 0), max_elec)
                if printing:
                    print(f"i={i}, t={t}, hod={hour_of_day}")
                    print(f"planned={remaining_planned}, produced={daily_count}")
                    print(f"surplus={surplus}, wanted={wanted}")
                    print(f"Original prod: {h_prod}, Resulting prod: {h_adj}")

        daily_count += h_adj
        settlementd = realized[t] - forward_bid - h_adj
        up = np.maximum(-settlementd, 0)
        dw = np.maximum(settlementd, 0)
        obj = (
                forward_bid * prices_F[t]
                + price_H * h_adj
                + dw * prices_S[t]
                - up * prices_B[t]
                - missing_production * penalty
        )

        forward_bids.append(forward_bid)
        ds.append(d)
        h_prods.append(h_adj)
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


def apply_risky_policy(results_to_copy, idx_start, idx_end, printing=False):
    """
    Applies a risky policy for adjusting production based on given results.

    Args:
        results_to_copy (dict): Results to copy and adjust.
        idx_start (int): Start index.
        idx_end (int): End index.
        printing (bool, optional): If True, print additional information. Defaults to False.

    Returns:
        dict: Adjusted results.
    """
    results = copy.deepcopy(results_to_copy)
    (prices_B, prices_S, prices_F, prices_forecast, features, features_red, realized, price_H, max_elec, nominal_wind,
     penalty, min_production, forecast_production) = import_consts()

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
        hour_of_day = (i % 24)
        if (hour_of_day == 0) and t != idx_start:
            missing_production = np.maximum(min_production - daily_count, 0)
            daily_count = 0

        forward_bid = results['forward_bid'][i]
        h_prod = results['h_prod'][i]

        d = realized[t] - forward_bid

        opt_h = get_hydro_opt(d, price_H, prices_S[t], prices_B[t], max_elec)

        remaining_hours = 23 - hour_of_day
        if hour_of_day == 23:
            remaining_planned = 0
        else:
            remaining_planned = np.sum([results['h_prod'][i + j] for j in range(1, remaining_hours)])

        if opt_h > h_prod:
            h_prod = opt_h
        else:
            surplus = daily_count + remaining_planned - min_production
            wanted = h_prod - opt_h
            if surplus >= wanted or hour_of_day == 23:
                h_adj = opt_h
            else:
                forward_prices_remaining = copy.deepcopy(prices_F[t:t + (23 - hour_of_day)])
                while wanted > 0:
                    preffered_idx = np.argmin(forward_prices_remaining)
                    price_to_remove = forward_prices_remaining[preffered_idx]
                    if prices_B[t] < price_to_remove:
                        h_adj = np.minimum(np.maximum(h_prod - surplus, 0), max_elec)
                        break
                    else:
                        free_turn_up = max_elec - results['h_prod'][i + preffered_idx]
                        if free_turn_up >= wanted:
                            results['h_prod'][i + preffered_idx] += wanted
                            h_adj = opt_h
                            break
                        else:
                            results['h_prod'][i + preffered_idx] = max_elec
                            wanted -= free_turn_up
                            forward_prices_remaining[preffered_idx] = 9999
                            remaining_planned = 0 if hour_of_day == 23 else np.sum(
                                [results['h_prod'][i + j] for j in range(1, remaining_hours)])
                            surplus = daily_count + remaining_planned - min_production

        daily_count += h_prod
        settlementd = realized[t] - forward_bid - h_prod
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