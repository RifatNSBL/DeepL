import numpy as np
import copy

class NeuralNetwork():
    def __init__(self, optimizer, weight_initializer, bias_initializer):
        self.optimizer = optimizer 
        self.loss = list()            
        self.layers = list()           
        self.data_layer = None      
        self.loss_layer = None
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

    def forward(self):
        input_tensor, self.label_tensor = copy.deepcopy(self.data_layer.next()) 
        """
        A deep copy means that all levels of the object are copied recursively, 
        ensuring that changes to the new object do not affect the original object and vice versa.
        """
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        
        output = self.loss_layer.forward(input_tensor, copy.deepcopy(self.label_tensor))
        return output
    
    def backward(self):
        y = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            y = layer.backward(y)
    
    def append_layer(self, layer):
        if layer.trainable:
            # layer.weights = self.weight_initializer.initialize()
            # layer.biases = self.bias_initializer.initialize()
            layer.initialize(self.weight_initializer, self.bias_initializer)
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self, iterations):
        for iteration in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

    def test(self, input_tensor):
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor