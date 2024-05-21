import numpy as np
from .Base import BaseLayer


class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.nextIn = input_tensor
        self.nextOut = np.maximum(0, input_tensor)
        return self.nextOut
    
    def backward(self, error_tensor):
        return np.where(self.nextIn > 0, error_tensor, 0)
