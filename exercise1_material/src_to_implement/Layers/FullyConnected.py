import numpy as np
from .Base import BaseLayer
import typing

class FullyConnected(BaseLayer):
    def __init__(self, input_size : int, output_size : int):
        super().__init__()
        self.trainable = True
        self._optimizer = None
        
        self.input_data = None
        self.input_size = input_size # represents number of columns, or the number of inputs of current layer
        self.output_size = output_size # represents number of rows, or the number of inputs of next layer
        self.weights = np.random.rand(self.input_size + 1, self.output_size)
        self.gradients = None#
        self.output_data = None

    def forward(self, input_tensor):
        bias = np.ones((input_tensor.shape[0], 1))
        self.input_data = np.hstack((input_tensor, bias))
        self.output_data = np.dot(self.input_data, self.weights)
        return self.output_data
    
    def backward(self, error_tensor : np.ndarray):
        self.gradient_weights = np.dot(self.input_data.T, error_tensor)
        self.gradient_bias = np.dot(self.weights, error_tensor.T)
        print("grad bias shape : ", self.gradient_bias.shape)
        print("grad weights shape : ", self.gradient_weights.shape)
        

        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        
        error_output = np.dot(error_tensor, self.weights.T)
        return error_output[:, :-1]


    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
