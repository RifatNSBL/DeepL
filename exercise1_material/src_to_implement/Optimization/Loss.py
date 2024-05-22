import typing
import numpy as np

class CrossEntropyLoss():
    def __init__(self):
        self.input_data = None
        self.output_data = None

    def forward(self, prediction_tesor : np.ndarray, label_tensor : np.ndarray):
        self.input_data = prediction_tesor
        eps = np.finfo(float).eps 
        self.output_data = -np.sum( np.multiply( np.log( self.input_data + eps), label_tensor ) )
        return self.output_data
    
    def backward(self, label_tensor : np.ndarray):
        eps = np.finfo(float).eps
        error_output = - ( label_tensor / (self.input_data + eps) )
        return error_output