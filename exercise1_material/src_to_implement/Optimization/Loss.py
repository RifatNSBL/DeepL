import numpy as np

class CrossEntropyLoss():
    def __init__(self):
        super().__init__()

    def forward(self, prediction_tensor, label_tensor):
        self.nextIn = prediction_tensor
        y_hat = prediction_tensor
        y = label_tensor
        loss = -np.sum(np.log(y_hat + np.finfo(float).eps) * y)
        return loss
    
    def backward(self, label_tensor):
        Entropy = - (label_tensor / self.nextIn + np.finfo(float).eps)
        return Entropy