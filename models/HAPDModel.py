from gurobipy import Model, GRB, quicksum

from models.TrainableModel import TrainableModel
from utils.constants import *
from utils.utils import get_forward_24, get_hydro_24


class HAPDModel(TrainableModel):

    def __init__(self, name, test_start_index=12 * HOURS_PER_MONTH, datafile=DATAFILE, nominal_wind=NOMINAL_WIND,
                 max_wind=NOMINAL_WIND, p_h_max=P_H_MAX,
                 h_min=H_MIN):
        super().__init__(name, test_start_index, datafile=datafile, nominal_wind=nominal_wind, max_wind=max_wind,
                         p_h_max=p_h_max,
                         h_min=h_min)

    def compute_optimal_weights(self, training_length):

        offset = self.test_start_index - training_length

        if training_length % 24 != 0:
            raise Exception("Training period must be a multiple of 24 hours!")

        # Declare Gurobi model
        initial_plan = Model()

        # Definition of variables
        settlements = initial_plan.addMVar(training_length, lb=-GRB.INFINITY, name="settlements")
        qF = initial_plan.addMVar((3, self.n_features + 1, HOURS_PER_DAY), lb=-GRB.INFINITY,
                                  name="qF")
        qH = initial_plan.addMVar((3, self.n_features + 1, HOURS_PER_DAY), lb=-GRB.INFINITY,
                                  name="qH")
        hydrogen_productions = initial_plan.addMVar(training_length, lb=0, name="hydrogen_productions", ub=self.p_h_max)
        forward_bids = initial_plan.addMVar(training_length, lb=-self.p_h_max, ub=self.max_wind,
                                            name="forward_bids")

        # Maximize profit
        initial_plan.setObjective(
            quicksum(
                self.prices_F[t + offset] * forward_bids[t]
                + PRICE_H * hydrogen_productions[t]
                + self.single_balancing_prices[t + offset] * settlements[t]
                for t in range(training_length)
            ),
            GRB.MAXIMIZE
        )

        # Constraints
        initial_plan.addConstrs(
            (self.realized[t + offset] - forward_bids[t] - hydrogen_productions[t] == settlements[t] for t in
             range(training_length)),
            name="settlement")

        for day in range(training_length // HOURS_PER_DAY):
            day_hours = list(range(HOURS_PER_DAY * day, HOURS_PER_DAY * (day + 1)))
            initial_plan.addConstr(quicksum(hydrogen_productions[t] for t in day_hours) >= self.h_min,
                                   name="min_hydrogen")
            for t in day_hours:
                index = t % 24
                if self.prices_F[t + offset] < PRICE_H:
                    initial_plan.addConstr(
                        forward_bids[t] == quicksum(
                            qF[0, i, index] * self.x.iloc[t + offset, i] for i in range(self.n_features)) +
                        qF[0, self.n_features, index], name="forward_bids")
                    initial_plan.addConstr(
                        hydrogen_productions[t] == quicksum(
                            qH[0, i, index] * self.x.iloc[t + offset, i] for i in range(self.n_features)) + qH[
                            0, self.n_features, index], name="hydrogen_productions")
                elif self.prices_F[t + offset] < TOP_DOMAIN:
                    initial_plan.addConstr(
                        forward_bids[t] == quicksum(
                            qF[1, i, index] * self.x.iloc[t + offset, i] for i in range(self.n_features)) +
                        qF[1, self.n_features, index], name="forward_bids")
                    initial_plan.addConstr(
                        hydrogen_productions[t] == quicksum(
                            qH[1, i, index] * self.x.iloc[t + offset, i] for i in range(self.n_features)) + qH[
                            1, self.n_features, index], name="hydrogen_productions")
                else:
                    initial_plan.addConstr(
                        forward_bids[t] == quicksum(
                            qF[2, i, index] * self.x.iloc[t + offset, i] for i in range(self.n_features)) +
                        qF[2, self.n_features, index], name="forward_bids")
                    initial_plan.addConstr(
                        hydrogen_productions[t] == quicksum(
                            qH[2, i, index] * self.x.iloc[t + offset, i] for i in range(self.n_features)) + qH[
                            2, self.n_features, index], name="hydrogen_productions")

        initial_plan.setParam('OutputFlag', 0)
        initial_plan.setParam('NumericFocus', 1)

        initial_plan.optimize()

        qF_values = np.array(
            [[qF[d, i, h].X for h in range(HOURS_PER_DAY)] for i in range(self.n_features + 1) for d in range(3)])
        qH_values = np.array(
            [[qH[d, i, h].X for h in range(HOURS_PER_DAY)] for i in range(self.n_features + 1) for d in range(3)])
        return qF_values, qH_values

    def train(self, training_length=12 * HOURS_PER_MONTH, save=True):

        qF_weights, qH_weights = self.compute_optimal_weights(training_length)

        data = np.concatenate([qF_weights, qH_weights], axis=0)

        names = np.concatenate([
            [f"qF{d + 1}_{i + 1}" for i in range(self.n_features + 1) for d in range(3)],
            [f"qH{d + 1}_{i + 1}" for i in range(self.n_features + 1) for d in range(3)]
        ])

        dataframe = pd.DataFrame(data.T, columns=names, index=range(HOURS_PER_DAY))
        self.weights = dataframe
        self.trained = True
        if save:
            self.save_weights()

    def get_schedule_from_weights(self, schedule_length, fm=False):

        forward_bids = []
        h_prods = []
        for t in range(self.test_start_index, self.test_start_index + schedule_length):
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

        dataframe = pd.DataFrame([forward_bids, h_prods]).T
        dataframe.columns = ["forward_bid", "hydrogen_production"]

        return dataframe
