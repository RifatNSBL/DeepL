import typing
import numpy as np
from .Base import BaseLayer

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_data = None
        self.output_data = None

    def forward(self, input_tensor):
        max_arg = np.max(input_tensor)
        self.input_data = np.subtract(input_tensor, max_arg)
        exp_data = np.exp(self.input_data)
        self.output_data = exp_data / np.sum(exp_data, axis=1)[:, np.newaxis] # np.sum removes axis, nned to add it back
        return self.output_data
    
    def backward(self, error_tensor : np.ndarray):
        vector = np.sum(np.multiply(error_tensor, self.output_data), axis=1)[:, np.newaxis] # here np.newaxis was in wrong position
        error_output = np.multiply(self.output_data, error_tensor - vector) 
        return error_output
