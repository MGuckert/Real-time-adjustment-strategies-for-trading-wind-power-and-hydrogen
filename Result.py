import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Result:
    def __init__(self, forward_bids, deviations, hydrogen_productions, settlements, missing_productions, objectives):
        self.forward_bids = forward_bids
        self.deviations = deviations
        self.hydrogen_productions = hydrogen_productions
        self.settlements = settlements
        self.missing_productions = missing_productions
        self.objectives = objectives
        self.forward_bids = forward_bids

    def get_total_objective(self) -> float:
        return np.sum(self.objectives)

    def get_total_hydrogen_production(self) -> float:
        return np.sum(self.hydrogen_productions)

    def get_average_hydrogen_production(self) -> float:
        return np.mean(self.hydrogen_productions)

    def get_average_hydrogen_production_per_day(self) -> float:
        return np.mean(
            [np.sum(self.hydrogen_productions[i:i + 24]) for i in range(0, len(self.hydrogen_productions), 24)])

    def plot_hydrogen_production(self):
        plt.figure(figsize=(15, 5))
        plt.plot(self.hydrogen_productions)
        plt.xlabel('Hour')
        plt.ylabel('Hydrogen Production')
        plt.title('Hydrogen Production over Time')
        plt.show()

    def plot_hydrogen_production_per_day_histogram(self):
        plt.figure(figsize=(15, 5))
        plt.hist(
            [np.sum(self.hydrogen_productions[i:i + 24]) for i in range(0, len(self.hydrogen_productions), 24)], bins=20)
        plt.xlabel('Frequency')
        plt.ylabel('Hydrogen Production')
        plt.title('Hydrogen Production per Day')
        plt.show()

    def plot_forward_bids(self):
        if self.forward_bids is None:
            raise ValueError("Forward bids data is not available.")
        plt.figure(figsize=(15, 5))
        plt.plot(self.forward_bids)
        plt.xlabel('Hour')
        plt.ylabel('Forward Bids')
        plt.title('Forward Bids over Time')
        plt.show()

    @staticmethod
    def plot_objectives(results, model_names, fig_title, save_fig=False, fig_name=None):
        original_obj = [result.get_total_objective() for result in results]
        original_adj_obj = [result.get_total_objective() - result.get_total_objective() for result in results]
        mpc_adj_obj = [result.get_total_objective() - result.get_total_objective() for result in results]
        local_hindsight_obj = [result.get_total_objective() if len(results) > 3 else None for result in results]
        bar_width = 0.4
        adjust_width = bar_width / 2
        r1 = np.arange(len(original_obj))
        colors = sns.color_palette("deep", 4)
        plt.bar(r1, original_obj, color=colors[0], width=bar_width, label='Day-ahead')
        plt.bar(r1 - adjust_width / 2, original_adj_obj, bottom=original_obj, color=colors[2], width=adjust_width,
                label='Adjustment rule')
        plt.bar(r1 + adjust_width / 2, mpc_adj_obj, bottom=original_obj, color=colors[1], width=adjust_width,
                label='MPC adjustment')
        plt.axhline(y=original_obj[0], color='black', linestyle='--', label='Global hindsight')

        if local_hindsight_obj is not None:
            local_hindsight_label_added = False
            # Add a line that is the size of the bar to represent the local hindsight for each model
            for i, obj in enumerate(local_hindsight_obj):
                xval = r1[i] - bar_width / 2  # Start of the bar
                width = bar_width  # Width of the bar
                yval = obj  # Top of the bar including local hindsight value
                if not local_hindsight_label_added:
                    plt.plot([xval, xval + width], [yval, yval], color=colors[3], linestyle=':', linewidth=2,
                             label='Local hindsight')
                    local_hindsight_label_added = True
                else:
                    plt.plot([xval, xval + width], [yval, yval], color=colors[3], linestyle=':', linewidth=2)

        plt.xlabel('Model', fontweight='bold')
        plt.ylabel('Objective Value', fontweight='bold')
        plt.xticks(r1, model_names)
        plt.legend()
        plt.title(fig_title)
        plt.ylim(min(original_obj) * 0.95, np.max(
            [np.array(original_obj) + np.array(original_adj_obj),
             np.array(original_obj) + np.array(mpc_adj_obj)]) * 1.05)
        plt.show()

        if save_fig:
            if fig_name is None:
                print("No figure name provided, saving as 'objectives_plot.png' by default.")
                fig_name = "objectives_plot.png"
            plt.savefig(fig_name)

    @staticmethod
    def get_mao(results):
        return [np.sum(results[3].objectives) - np.sum(results[i].objectives) for i in range(len(results) - 1)]
