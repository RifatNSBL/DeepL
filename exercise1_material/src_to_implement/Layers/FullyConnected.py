import numpy as np
from Layers import Base
from Optimization import Optimizers

class FullyConnected(Base.BaseLayer):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.weights = np.random.uniform(size=(input_size+1, output_size))
        self.bias = np.random.uniform(size=(1, output_size))
        self._optimizers = None


    def forward(self, input_tensor):
        self.x0 = np.ones((input_tensor.shape[0],1))
        self.fistLayer = np.hstack((input_tensor, self.x0))
        # self.fistLayer = input_tensor
        self.lastLayer = np.dot(self.fistLayer, self.weights)
        return self.lastLayer
    
    @property
    def optimizers(self):
        return self._optimizers
    
    @optimizers.setter
    def optimizers(self, optimizers):
        self._optimizers = optimizers
    

    def backward(self, error_tensor):
        self.gradient_bias = np.dot(self.weights, error_tensor.T)
        grad_W = np.dot(self.fistLayer.T, error_tensor)
        self.gradient_weights = grad_W
        
        if self._optimizers != None:
            self.weights = self._optimizers.calculate_update(self.weights, grad_W)
        
        error_output = np.dot(error_tensor, self.weights.T)
        return error_output[:,:-1]
