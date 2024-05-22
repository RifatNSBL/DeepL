import numpy as np
import copy

class NeuralNetwork():
    def __init__(self, optimizer):
        self.optimizer = optimizer 
        self.loss = list()            
        self.layers = list()           
        self.data_layer = None      
        self.loss_layer = None

    def forward(self):
        input_data, self.label_tensor = self.data_layer.next() 
        for layer in self.layers:
            input_data = layer.forward(input_data)

        output_data = self.loss_layer.forward(input_data, self.label_tensor)
        return output_data

    def backward(self):
        backprop_layer = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            backprop_layer = layer.backward(backprop_layer)

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self, iterations):
        for iteration in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

    def test(self, input_data):
        result = input_data
        for layer in self.layers:
            result = layer.forward(result)
        return result