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
        self._gradient_weights = None
        self._gradient_bias = None
        self._optimizer = None
        self._bias_optimizer = None
        self.forwad_shape = None

        self.weights = np.random.rand(self.num_kernals, *self.convolution_shape)
        self.bias = np.random.rand(self.num_kernals)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.batch_size = input_tensor.shape[0]
        self.channel_size = input_tensor.shape[1]

        if len(self.convolution_shape) == 2:
            c, m = self.convolution_shape
            self.height = self.input_tensor.shape[2]
            self.stride_h = self.stride_shape[0]
            self.output_h = int(np.ceil(self.height / self.stride_h))
            self.output_shape = (self.batch_size, self.num_kernals, self.output_h)
        else:
            c, m, n = self.convolution_shape
            self.height = self.input_tensor.shape[2]
            self.width = self.input_tensor.shape[3]
            self.stride_h = self.stride_shape[0]
            self.stride_w = self.stride_shape[1]
            self.output_h = int(np.ceil(self.height / self.stride_h))
            self.output_w = int(np.ceil(self.width / self.stride_w))
            self.output_shape = (self.batch_size, self.num_kernals, self.output_h, self.output_w)

        output = np.zeros(self.output_shape)

        for b in range(self.batch_size):
            for k in range(self.num_kernals): 
                conv = []
                for c in range(self.channel_size):

                    """
                    Often cross correlation is used in the forward pass
                    It must be mode='same' based on tutorial and my hand calculation draft
                    """
                    conv.append(signal.correlate(self.input_tensor[b,c], self.weights[k,c], mode='same')) 

                conv_stacked = np.stack(conv, axis=0)
                conv_sum = np.sum(conv_stacked, axis=0)

                if len(self.convolution_shape) == 2:
                    conv_sum = conv_sum[::self.stride_h]
                else:
                    conv_sum = conv_sum[::self.stride_h, ::self.stride_w]

                # print('conv_sum: ', conv_sum.shape, '\n')
                # print('bias: ', self.bias.shape, '\n')
                output[b, k] = conv_sum + self.bias[k]  # like wx + b 
        return output

    def backward(self, error_tensor):
        grad_in = np.zeros_like(self.input_tensor)

        #padding 
        pad_y_1 = self.weights.shape[2] // 2
        if self.weights.shape[2] % 2 == 0:  # even filter length -> non-symmetric padding required
            pad_y_2 = pad_y_1 - 1  # shorter padding at end of this dim
        else:
            pad_y_2 = pad_y_1
        if len(self.weights.shape[2:]) > 1:  # 2D kernel
            pad_x_1 = self.weights.shape[3] // 2
            if self.weights.shape[3] % 2 == 0:
                pad_x_2 = pad_x_1 - 1  # shorter padding at end of this dim
            else:
                pad_x_2 = pad_x_1
        

        if len(self.convolution_shape) == 2:
            new_error = np.zeros((error_tensor.shape[0], error_tensor.shape[1], self.height))
            new_error[:,:, ::self.stride_h] = error_tensor
        else:
            new_error = np.zeros((error_tensor.shape[0], error_tensor.shape[1], self.height, self.width))
            new_error[:,:, ::self.stride_h, ::self.stride_w] = error_tensor

        self._gradient_weights = np.zeros(self.weights.shape)
        self._gradient_bias = np.zeros(self.bias.shape)

        for b in range(self.batch_size):
            input = self.input_tensor[b,:]
            for k in range(self.num_kernals):
                for c in range(self.channel_size):
                    grad_in[b, c] += signal.convolve(new_error[b, k], self.weights[k, c], mode='same')

                    if len(self.weights.shape[2:]) > 1:
                        padded_input = np.pad(input[c,:], ((pad_y_1, pad_y_2), (pad_x_1, pad_x_2)),
                                        mode='constant', constant_values=0)
                    else:
                        padded_input = np.pad(input[c,:], (pad_y_1, pad_y_2),
                                        mode='constant', constant_values=0)


                    self._gradient_weights[k, c] += signal.correlate(padded_input, new_error[b, k], mode='valid')
                self._gradient_bias[k] += np.sum(new_error[b, k])

        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)
        if self._bias_optimizer:
            self.bias = self._bias_optimizer.calculate_update(self.bias, self._gradient_bias)

        return grad_in
    
    def initialize(self, weights_initializer, bias_initializer):
        fan_in = np.prod(self.convolution_shape)
        fan_out = self.num_kernals * np.prod(self.convolution_shape[1:])
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, 1, self.num_kernals)

    @property 
    def gradient_weights(self):
        return self._gradient_weights
    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    @property
    def gradient_bias(self):
        return self._gradient_bias
    @gradient_bias.setter
    def gradient_bias(self, value):
        self._gradient_bias = value

    @property
    def optimizer(self):
        return self._optimizer
    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value
