import numpy as np
from .Base import BaseLayer

class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tesnor):
        self.input_shape = input_tesnor.shape
        return input_tesnor.reshape(input_tesnor.shape[0], -1)

    def backward(self, error_tensor):
        return error_tensor.reshape(self.input_shape)