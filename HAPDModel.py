import overrides

from BaseModel import BaseModel
from constants import *
from gurobipy import Model, GRB, quicksum
import numpy as np

from evaluation import get_forward_24, get_hydro_24


class HAPDModel(BaseModel):
    # Write the docstring

    """
    HAPDModel class for the Hourly Architecture, 3 Price Domains Model.

    Args:
        filename (str): Name of the file.
        nominal_wind (float): Nominal wind power.
        max_wind (float): Maximum wind power.
        p_h_max (float): Maximum hydrogen production.
        h_min (float): Minimum hydrogen production.

    Attributes:
        filename (str): Name of the file.
        nominal_wind (float): Nominal wind power.
        max_wind (float): Maximum wind power.
        p_h_max (float): Maximum hydrogen production.
        h_min (float): Minimum hydrogen production.
        weights (DataFrame): Weights of the model.
        trained (bool): Flag for training.
        results (dict): Results of the evaluation.
        evaluated (bool): Flag for evaluation.

    Methods:
        get_initial_plan(training_period_length, bidding_start): Get the initial plan.
        train(training_length, bidding_start, save): Train the model.
        evaluate(idx_start, idx_end, fm): Evaluate the model.
        get_schedule(idx_start, idx_end, fm): Get the schedule.
        summary(): Print the summary.
        get_results(): Get the results.

    """

    def __init__(self, filename, datafile=DATAFILE, nominal_wind=NOMINAL_WIND, max_wind=NOMINAL_WIND, p_h_max=P_H_MAX, h_min=H_MIN):
        super().__init__(filename, datafile, nominal_wind=nominal_wind, max_wind=max_wind, p_h_max=p_h_max, h_min=h_min)
        self.summary()

    def get_initial_plan(self, training_period_length, bidding_start):

        print(f"Prices: {self.prices_B[bidding_start]}, {self.prices_S[bidding_start]}, {self.prices_F[bidding_start]}")
        print(f"Realized: {self.realized[bidding_start]}")

        offset = bidding_start - training_period_length

        if training_period_length % 24 != 0:
            raise Exception("Training period must be a multiple of 24 hours!")

        # Declare Gurobi model
        initial_plan = Model()

        # Definition of variables
        E_DW = initial_plan.addMVar(training_period_length, lb=0, name="E_DW")
        E_UP = initial_plan.addMVar(training_period_length, lb=0, name="E_UP")
        b = initial_plan.addMVar(training_period_length, vtype=GRB.BINARY, name="b")
        E_settle = initial_plan.addMVar(training_period_length, lb=-GRB.INFINITY, name="E_settle")
        qF = initial_plan.addMVar((3, self.n_features + 1, HOURS_PER_DAY), lb=-GRB.INFINITY,
                                  name="qF")
        qH = initial_plan.addMVar((3, self.n_features + 1, HOURS_PER_DAY), lb=-GRB.INFINITY,
                                  name="qH")
        hydrogen = initial_plan.addMVar(training_period_length, lb=0, name="hydrogen", ub=self.p_h_max)
        forward_bid = initial_plan.addMVar(training_period_length, lb=-self.p_h_max, ub=self.max_wind,
                                           name="forward_bid")

        # Maximize profit
        initial_plan.setObjective(
            quicksum(
                self.prices_F[t + offset] * forward_bid[t]
                + PRICE_H * hydrogen[t]
                + self.prices_S[t + offset] * E_DW[t]
                - self.prices_B[t + offset] * E_UP[t]
                for t in range(training_period_length)
            ),
            GRB.MAXIMIZE
        )

        # Constraints
        initial_plan.addConstrs(
            (self.realized[t + offset] - forward_bid[t] - hydrogen[t] == E_settle[t] for t in
             range(training_period_length)),
            name="settlement")
        initial_plan.addConstrs((E_DW[t] >= E_settle[t] for t in range(training_period_length)), name="surplus_settle1")
        initial_plan.addConstrs((E_DW[t] <= E_settle[t] + self.M * b[t] for t in range(training_period_length)),
                                name="surplus_settle2")
        initial_plan.addConstrs((E_DW[t] <= self.M * (1 - b[t]) for t in range(training_period_length)),
                                name="surplus_settle3")
        initial_plan.addConstrs((E_UP[t] >= -E_settle[t] for t in range(training_period_length)),
                                name="deficit_settle1")
        initial_plan.addConstrs((E_UP[t] <= -E_settle[t] + self.M * (1 - b[t]) for t in range(training_period_length)),
                                name="deficit_settle2")
        initial_plan.addConstrs((E_UP[t] <= self.M * b[t] for t in range(training_period_length)),
                                name="deficit_settle3")

        for day in range(training_period_length // HOURS_PER_DAY):
            day_hours = list(range(HOURS_PER_DAY * day, HOURS_PER_DAY * (day + 1)))
            initial_plan.addConstr(quicksum(hydrogen[t] for t in day_hours) >= self.h_min, name="min_hydrogen")
            for t in day_hours:
                index = t % 24
                if self.prices_F[t + offset] < PRICE_H:
                    initial_plan.addConstr(
                        forward_bid[t] == quicksum(
                            qF[0, i, index] * self.x.iloc[t + offset, i] for i in range(self.n_features)) +
                        qF[0, self.n_features, index], name="forward_bid")
                    initial_plan.addConstr(
                        hydrogen[t] == quicksum(
                            qH[0, i, index] * self.x.iloc[t + offset, i] for i in range(self.n_features)) + qH[
                            0, self.n_features, index], name="hydrogen")
                elif self.prices_F[t + offset] < TOP_DOMAIN:
                    initial_plan.addConstr(
                        forward_bid[t] == quicksum(
                            qF[1, i, index] * self.x.iloc[t + offset, i] for i in range(self.n_features)) +
                        qF[1, self.n_features, index], name="forward_bid")
                    initial_plan.addConstr(
                        hydrogen[t] == quicksum(
                            qH[1, i, index] * self.x.iloc[t + offset, i] for i in range(self.n_features)) + qH[
                            1, self.n_features, index], name="hydrogen")
                else:
                    initial_plan.addConstr(
                        forward_bid[t] == quicksum(
                            qF[2, i, index] * self.x.iloc[t + offset, i] for i in range(self.n_features)) +
                        qF[2, self.n_features, index], name="forward_bid")
                    initial_plan.addConstr(
                        hydrogen[t] == quicksum(
                            qH[2, i, index] * self.x.iloc[t + offset, i] for i in range(self.n_features)) + qH[
                            2, self.n_features, index], name="hydrogen")

        initial_plan.setParam('NumericFocus', 1)

        initial_plan.optimize()

        qF_values = np.array(
            [[qF[d, i, h].X for h in range(HOURS_PER_DAY)] for i in range(self.n_features + 1) for d in range(3)])
        qH_values = np.array(
            [[qH[d, i, h].X for h in range(HOURS_PER_DAY)] for i in range(self.n_features + 1) for d in range(3)])
        return qF_values, qH_values

    def train(self, training_length=12 * HOURS_PER_MONTH, bidding_start=12*HOURS_PER_MONTH, save=True):

        qF_weights, qH_weights = self.get_initial_plan(training_length, bidding_start)

        data = np.concatenate([qF_weights, qH_weights], axis=0)

        names = np.concatenate([
            [f"qF{d + 1}_{i + 1}" for i in range(self.n_features + 1) for d in range(3)],
            [f"qH{d + 1}_{i + 1}" for i in range(self.n_features + 1) for d in range(3)]
        ])

        dataframe = pd.DataFrame(data.T, columns=names, index=range(HOURS_PER_DAY))
        if save:
            self.save_weights(dataframe)
        self.weights = dataframe
        self.trained = True

    def get_schedule_from_weights(self, idx_start, idx_end, fm=False):

        forward_bids = []
        h_prods = []
        for t in range(idx_start, idx_end):
            if self.prices_F[t] < PRICE_H:
                domain = 1
            elif self.prices_F[t] < TOP_DOMAIN:
                domain = 2
            else:
                domain = 3

            if fm:
                df_f = self.weights[[f"qF{domain}_{i + 1}" for i in range(3)]]
                df_h = self.weights[[f"qH{domain}_{i + 1}" for i in range(3)]]
                forward_bid = get_forward_24(df_f, [self.forecasted_prod, self.prices_F], t, fm)
                h_prod = get_hydro_24(df_h, [self.forecasted_prod, self.prices_F], self.p_h_max, t, fm)
            else:
                df_f = self.weights[[f"qF{domain}_{i + 1}" for i in range(len(self.x.columns) + 1)]]
                df_h = self.weights[[f"qH{domain}_{i + 1}" for i in range(len(self.x.columns) + 1)]]
                forward_bid = get_forward_24(df_f, self.x, t, fm)
                h_prod = get_hydro_24(df_h, self.x, self.p_h_max, t, fm)

            forward_bid = np.minimum(forward_bid, self.p_h_max)
            forward_bid = np.maximum(forward_bid, -self.p_h_max)

            h_prod = np.maximum(h_prod, 0)
            h_prod = np.minimum(self.p_h_max, h_prod)

            forward_bids.append(forward_bid)
            h_prods.append(h_prod)

        return forward_bids, h_prods

    def evaluate(self, idx_start, idx_end, fm=False):
        """
        Perform complete evaluation for hourly models with price domains in a given time period.

        Args:
            df (DataFrame): Data frame containing quantities.
            idx_start (int): Start index.
            idx_end (int): End index.
            fm (bool): Flag for model.
            reduced_features (bool): Flag for reduced features.

        Returns:
            dict: Results of the evaluation.
        """

        if self.weights is None:
            try:
                self.load_weights()
            except FileNotFoundError:
                print("Model not trained. Please train the model first.")
                return

        forward_bids = []
        ds = []
        h_prods = []
        ups = []
        dws = []
        objs = []
        missing_productions = []
        missing_production = 0
        daily_count = 0

        for t in range(idx_start, idx_end):
            hour_of_day = (t % 24)
            if hour_of_day == 0 and t != idx_start:
                missing_production = np.maximum(self.h_min - daily_count, 0)
                daily_count = 0

            if self.prices_F[t] < PRICE_H:
                domain = 1
            elif self.prices_F[t] < TOP_DOMAIN:
                domain = 2
            else:
                domain = 3

            if fm:
                df_f = self.weights[[f"qF{domain}_{i + 1}" for i in range(3)]]
                df_h = self.weights[[f"qH{domain}_{i + 1}" for i in range(3)]]
                forward_bid = get_forward_24(df_f, [self.forecasted_prod, self.prices_F], t, fm)
                h_prod = get_hydro_24(df_h, [self.forecasted_prod, self.prices_F], self.p_h_max, t, fm)
            else:
                df_f = self.weights[[f"qF{domain}_{i + 1}" for i in range(len(self.x.columns) + 1)]]
                df_h = self.weights[[f"qH{domain}_{i + 1}" for i in range(len(self.x.columns) + 1)]]
                forward_bid = get_forward_24(df_f, self.x, t, fm)
                h_prod = get_hydro_24(df_h, self.x, self.p_h_max, t, fm)

            forward_bid = np.minimum(forward_bid, self.p_h_max)
            forward_bid = np.maximum(forward_bid, -self.p_h_max)
            d = self.realized[t] - forward_bid

            h_prod = np.maximum(h_prod, 0)
            h_prod = np.minimum(self.p_h_max, h_prod)
            daily_count += h_prod
            settlementd = self.realized[t] - forward_bid - h_prod
            up = np.maximum(-settlementd, 0)
            dw = np.maximum(settlementd, 0)
            obj = (
                    forward_bid * self.prices_F[t]
                    + PRICE_H * h_prod
                    + dw * self.prices_S[t]
                    - up * self.prices_B[t]
                    - missing_production * PENALTY
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
        self.results = results
        self.evaluated = True

    @overrides.override(BaseModel)
    def evaluate(self):