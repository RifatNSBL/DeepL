import typing
import numpy as np

class CrossEntropyLoss():
    def __init__(self):
        pass

    def forward(self, prediction_tesor : np.ndarray, label_tensor : np.ndarray):
        return np.sum(-(np.log()))