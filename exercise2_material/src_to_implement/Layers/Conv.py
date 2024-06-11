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
                    It must be mode='same' based on tutorial and my hand calculation draft
                    """
                    conv.append(signal.correlate(self.input_tensor[b,c], self.weights[k,c], mode='same')) 

                conv_stacked = np.stack(conv, axis=0)
                conv_sum = np.sum(conv_stacked, axis=0)

                self.forwad_shape = conv_sum.shape

                if len(self.convolution_shape) == 2:
                    conv_sum = conv_sum[::stride_h]
                else:
                    conv_sum = conv_sum[::stride_h, ::stride_w]

                # print('conv_sum: ', conv_sum.shape, '\n')
                # print('bias: ', self.bias.shape, '\n')
                output[b, k] = conv_sum + self.bias[k]  # like wx + b 
        return output

    def backward(self, error_tensor):
        # grad_in = np.zeros(self.input_tensor.shape)
        # new_error = np.zeros(self.forwad_shape)
        # stride_h = self.stride_shape[0]
        # stride_w = self.stride_shape[1]
        # if len(self.convolution_shape) == 2:
        # new_error[::stride_h] = error_tensor



        grad_in = np.zeros(self.input_tensor.shape)
        new_weights = np.copy(self.weights)

        if len(self.convolution_shape)==3:
            tmp_grad_weight = np.zeros((error_tensor.shape[0], self.weights.shape[0], self.weights.shape[1],
                                              self.weights.shape[2], self.weights.shape[3]))
            ##padding
            conv_output = []
            batch_size = self.input_tensor.shape[0]
            for batch in range(batch_size):
                out_tot = []
                kernels = self.input_tensor.shape[1]
                for kernel in range(kernels):
                    
                    height = self.convolution_shape[1]
                    width = self.convolution_shape[2]
                    pad_height = (height // 2, height // 2)
                    pad_width = (width // 2, width // 2)
                    
                    out_tot.append(np.pad(self.input_tensor[batch, kernel], (pad_height, pad_width), mode='constant'))

                    if self.convolution_shape[2]%2 ==0:
                        out_tot[kernel] = out_tot[kernel][:,:-1]
                    if self.convolution_shape[1]%2 ==0:
                        out_tot[kernel] = out_tot[kernel][:-1,:]

                conv_part = np.stack(out_tot, axis=0)
                
                conv_output.append(conv_part)
            padded_input = np.stack(conv_output, axis=0)

            batch_size = error_tensor.shape[0]
            for batch in range(batch_size):
                kernels = error_tensor.shape[1]
                for kernel in range(kernels):

                    temp = signal.resample(error_tensor[batch, kernel], error_tensor[batch, kernel].shape[0] * self.stride_shape[0], axis=0)
                    temp = signal.resample(temp, error_tensor[batch, kernel].shape[1] * self.stride_shape[1], axis=1)
                    temp = temp[:self.input_tensor.shape[2], :self.input_tensor.shape[3]]
                    mask_rows = np.arange(temp.shape[0]) % self.stride_shape[0] != 0
                    mask_cols = np.arange(temp.shape[1]) % self.stride_shape[1] != 0
                    temp[mask_rows, :] = 0
                    temp[:, mask_cols] = 0  

                    channels = self.input_tensor.shape[1]
                    for channel in range(channels):
                        tmp_grad_weight[batch, kernel, channel] = signal.correlate(padded_input[batch, channel], temp, mode='valid')
          
            self.gradient_weights = tmp_grad_weight.sum(axis=0)

        axes = tuple(range(len(new_weights.shape)))
        if len(self.convolution_shape) == 3:
            axes = axes[1], axes[0], axes[2], axes[3]
        elif len(self.convolution_shape) == 2:
            axes = axes[1], axes[0], axes[2]

        new_weights = np.transpose(new_weights, axes)
        
        batch_size= error_tensor.shape[0]
        for batch in range(batch_size):
            kernels =new_weights.shape[0]    
            for kernel in range(kernels):
                out_tot = []
                channels = new_weights.shape[1]
                for channel in range(channels):

                    if len(self.convolution_shape) == 3:
                        temp = signal.resample(error_tensor[batch, channel], error_tensor[batch, channel].shape[0] * self.stride_shape[0], axis=0)
                        temp = signal.resample(temp, error_tensor[batch, channel].shape[1] * self.stride_shape[1], axis=1)
                        temp = temp[:self.input_tensor.shape[2], :self.input_tensor.shape[3]]
                        mask_rows = np.arange(temp.shape[0]) % self.stride_shape[0] != 0
                        mask_cols = np.arange(temp.shape[1]) % self.stride_shape[1] != 0
                        temp[mask_rows, :] = 0
                        temp[:, mask_cols] = 0

                    elif len(self.convolution_shape) == 2:
                        temp = signal.resample(error_tensor[batch, channel], error_tensor[batch, channel].shape[0] * self.stride_shape[0], axis=0)
                        temp = temp[:self.input_tensor.shape[2]]
                        mask_rows = np.arange(temp.shape[0]) % self.stride_shape[0] != 0
                        temp[mask_rows] = 0

                    out_tot.append(signal.convolve(temp, new_weights[kernel, channel], mode='same', method='direct'))

                temp2 = np.stack(out_tot, axis=0)
                temp2 = temp2.sum(axis=0)
                grad_in[batch, kernel] = temp2

       
        if len(self.convolution_shape)==3:
            self.gradient_bias = np.sum(error_tensor, axis=(0,2,3))
        elif len(self.convolution_shape)==2:
            self.gradient_bias = np.sum(error_tensor, axis=(0,2))

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
    def optimizer(self):
        return self._optimizer
    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value
