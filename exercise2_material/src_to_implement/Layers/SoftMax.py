import numpy as np
from .Base import BaseLayer
from Optimization import *


class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.nextIn = input_tensor
        temp = np.exp(input_tensor - input_tensor.max(axis = 1)[np.newaxis].T)
        self.nextOut = temp / temp.sum(axis = 1)[np.newaxis].T
        return self.nextOut
    
    def backward(self, error_tensor):
        return self.nextOut * (error_tensor - (error_tensor * self.nextOut).sum(axis = 1)[np.newaxis].T)
    