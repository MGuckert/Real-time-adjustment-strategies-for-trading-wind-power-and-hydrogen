from abc import ABC, abstractmethod

from constants import *
from BaseModel import BaseModel


class TrainableModel(BaseModel, ABC):
    def __init__(self, name, test_start_index, datafile, nominal_wind, max_wind, p_h_max, h_min):
        super().__init__(name, test_start_index, datafile, nominal_wind, max_wind, p_h_max, h_min)
        self.weights = None
        self.trained = False
        self.summary()

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def get_schedule_from_weights(self, idx_end, fm):
        pass

    @abstractmethod
    def evaluate(self, eval_length):
        pass

    def save_weights(self, weights):
        print(os.path.join(RESULTS_DIR, f"{self.name}_weights.csv"))
        weights.to_csv(os.path.join(RESULTS_DIR, f"{self.name}_weights.csv"), index=False)
        print("Weights saved successfully")

    def load_from_weights(self):
        filepath = os.path.join(RESULTS_DIR, f"{self.name}_weights.csv")
        if not os.path.exists(filepath):
            raise FileNotFoundError("No saved weights found")
        self.weights = pd.read_csv(filepath)
        self.trained = True
        print("Weights loaded successfully")
