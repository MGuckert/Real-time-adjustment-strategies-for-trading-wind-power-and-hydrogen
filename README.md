# Real-time adjustment strategies for trading wind power and hydrogen

The new class-based code for the feature-driven models can be found in the models package.

By default, the data used by the models is the 2022-2023 data.

The models that were used and evaluated in the thesis are the following:

- Deterministic model
- Hindsight model
- HAPD model (the HAPD-AF-12 version)

These models, along with their evaluation and adjustment results, can be found in the results/model folder.

The adjustments currently considered are:

- No adjustment ("original")
- Best adjustment: best adjustment possible with full information
- Rule-based adjustment: adjustment based on a rule-based approach (hour by hour optimization)
- Naive MPC adjustment: MPC adjustment with naive balancing prices forecasts (previous day prices)
- Stochastic MPC adjustment: MPC adjustment where the objective corresponds to the expected value of the revenue using generated balancing prices scenarios.
