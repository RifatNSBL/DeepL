import numpy as np

class Sgd:
    def __init__(self, learning_rate:float):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        weight_tensor = weight_tensor - self.learning_rate * gradient_tensor
        return weight_tensor

class SgdWithMomentum:
    def __init__(self, learning_rate:float, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = None

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.v == None:
            self.v = np.zeros_like(weight_tensor)
            
        self.v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor
        return weight_tensor + self.v

class Adam:
    def __init__(self, learning_rate:float, mu:float, rho:float):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v = None
        self.r = None
        self.t = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.v == None:
            self.v = np.zeros_like(weight_tensor)
        if self.r == None:
            self.r = np.zeros_like(weight_tensor)

        self.t += 1

        self.v = self.mu * self.v + (1-self.mu) * gradient_tensor
        self.r = self.rho * self.r + (1-self.rho) * np.square(gradient_tensor)

        v_hat = self.v / (1 - np.power(self.mu, self.t))
        r_hat = self.r / (1 - np.power(self.rho, self.t))

        weight_update = self.learning_rate * v_hat / ((np.sqrt(r_hat)) + np.finfo(float).eps)

        return weight_tensor - weight_update