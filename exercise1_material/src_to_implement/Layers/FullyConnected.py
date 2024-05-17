import numpy as np
from Layers import Base
from Optimization import Optimizers

class FullyConnected(Base.BaseLayer):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.weights = np.random.uniform(size=(input_size + 1, output_size))
        self.bias = np.random.uniform(size=(1, output_size))
        self._optimizers = None


    def forward(self, input_tensor):
        self.x0 = np.ones((input_tensor.shape[0],1))
        input_tensor = np.hstack((self.x0 ,input_tensor))
        self.fistLayer = input_tensor
        self.lastLayer = np.dot(input_tensor, self.weights) # + self.bias
        return self.lastLayer
    
    @property
    def optimizers(self):
        return self._optimizers
    
    @optimizers.setter
    def optimizers(self, optimizers):
        self._optimizers = optimizers
    

    def backward(self, error_tensor):
        pre_error = np.dot(error_tensor, np.transpose(self.weights))
        grad_W = np.dot(np.transpose(self.fistLayer), error_tensor)

        if self._optimizers != None:
            self.weights = self._optimizers.calculate_update(self.weights, grad_W)
            # self.bias = self._optimizers.calculate_update(self.bias, error_tensor)
            
        # self.grad_bias = error_tensor
        self.gradient_weights = grad_W

        return pre_error[:,1:]
