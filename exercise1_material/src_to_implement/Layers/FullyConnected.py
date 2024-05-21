import numpy as np
from .Base import BaseLayer

class FullyConnected(BaseLayer):

    def __init__(self, input_size:int, output_size:int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True
        self.weights = np.random.rand(self.input_size + 1, self.output_size)
        self._optimizer = None

    def forward(self, input_tensor):
        x0 = np.ones((input_tensor.shape[0],1))
        self.fistLayer = np.hstack((input_tensor, x0))
        self.lastLayer = np.dot(self.fistLayer, self.weights)
        return self.lastLayer

    def backward(self, error_tensor:np.ndarray):
        self.gradient_bias = np.dot(self.weights, error_tensor.T)
        self.gradient_weights = np.dot(self.fistLayer.T, error_tensor)
        
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)
        
        error_output = np.dot(error_tensor, self.weights.T)
        return error_output[:, :-1]

    @property
    def optimizer(self):
        return self._optimizer
    
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
