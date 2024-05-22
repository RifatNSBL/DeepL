import typing
import numpy as np
from .Base import BaseLayer

class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_data = None
        self.output_data = None

    def forward(self, input_tensor):
        self.input_data = input_tensor
        self.output_data = np.maximum(0,self.input_data)
        return self.output_data
    
    def backward(self, error_tensor : np.ndarray):
        grad_func = np.vectorize(return_der)
        gradient = grad_func(self.output_data)
        error_output = np.multiply(error_tensor, gradient)
        return error_output

def return_der(x):
    return 1 if x > 0 else 0