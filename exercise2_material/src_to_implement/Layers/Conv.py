import numpy as np
from .Base import BaseLayer
from scipy import signal


class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernals):
        super().__init__()
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernals = num_kernals
        self.trainable = True
        self._gradient_weight = None

        self.weights = np.random.rand(self.num_kernals, *self.convolution_shape)
        self.bias = np.random.rand(self.num_kernals)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.batch_size = input_tensor.shape[0]
        self.channel_size = input_tensor.shape[1]

        if len(self.convolution_shape) == 2:
            c, m = self.convolution_shape
            height = self.input_tensor.shape[2]
            stride_h = self.stride_shape[0]
            output_h = int(np.ceil(height / stride_h))
            output_shape = (self.batch_size, self.num_kernals, output_h)
        else:
            c, m, n = self.convolution_shape
            height = self.input_tensor.shape[2]
            width = self.input_tensor.shape[3]
            stride_h = self.stride_shape[0]
            stride_w = self.stride_shape[1]
            output_h = int(np.ceil(height / stride_h))
            output_w = int(np.ceil(width / stride_w))
            output_shape = (self.batch_size, self.num_kernals, output_h, output_w)

        output = np.zeros(output_shape)

        for b in range(self.batch_size):
            for k in range(self.num_kernals): 
                conv = []
                for c in range(self.channel_size):

                    """
                    Often cross correlation is used in the forward pass
                    It must be mode='same' based on my hand calculation draft
                    """
                    conv.append(signal.correlate(self.input_tensor[b,c], self.weights[k,c], mode='same')) 

                conv_stacked = np.stack(conv, axis=0)
                conv_sum = np.sum(conv_stacked, axis=0)

                if len(self.convolution_shape) == 2:
                    conv_sum = conv_sum[::stride_h]
                else:
                    conv_sum = conv_sum[::stride_h, ::stride_w]

                # print('conv_sum: ', conv_sum.shape, '\n')
                # print('bias: ', self.bias.shape, '\n')
                output[b, k] = conv_sum + self.bias[k]  # like wx + b 
        return output

    def backward(self, error_tensor):
        pass


@property 
def gradient_weights(self):
    return self._gradient_weight

@gradient_weights.setter
def gradient_weight(self, value):
    self._gradient_weight = value
