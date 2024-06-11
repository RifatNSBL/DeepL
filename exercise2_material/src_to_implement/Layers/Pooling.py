from .Base import BaseLayer
import numpy as np


class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape

    def forward(self, input_tensor):
        _y = (input_tensor.shape[2] - self.pooling_shape[0] ) // self.stride_shape[0] + 1
        _x = (input_tensor.shape[3] - self.pooling_shape[1] ) // self.stride_shape[1] + 1
        output_init = np.zeros((input_tensor.shape[0], input_tensor.shape[1], _y, _x))

        self.input_shape = input_tensor.shape
        self.max_mask = np.zeros((output_init.shape[0],output_init.shape[1],
                                 output_init.shape[2],output_init.shape[3],
                                 self.pooling_shape[0], self.pooling_shape[1]))

        for b in range(input_tensor.shape[0]):
            for c in range(input_tensor.shape[1]):
                yy = 0
                for y in range(0, input_tensor.shape[2], self.stride_shape[0]):
                    if y + self.pooling_shape[0] > input_tensor.shape[2]: continue
                    xx = 0
                    for x in range(0, input_tensor.shape[3], self.stride_shape[1]):
                        if x+self.pooling_shape[1] > input_tensor.shape[3]: continue
                        pool_window = input_tensor[b, c, y:y+self.pooling_shape[0], x:x+self.pooling_shape[1]]
                        max_val = np.max(pool_window)
                        output_init[b, c, yy, xx] = max_val
                        self.max_mask[b, c, yy, xx] = pool_window == max_val
                        xx += 1
                    yy += 1

        return output_init

    def backward(self, error_tensor):
        output = np.zeros(self.input_shape)

        for b in range(output.shape[0]):
            for c in range(output.shape[1]):
                yy = 0
                for y in range(0, output.shape[2], self.stride_shape[0]):
                    if y + self.pooling_shape[0] > output.shape[2]: continue
                    xx = 0
                    for x in range(0, output.shape[3], self.stride_shape[1]):
                        if x + self.pooling_shape[1] > output.shape[3]: continue
                        mask_window = self.max_mask[b, c, yy, xx]
                        output[b, c, y:y+self.pooling_shape[0], x:x+self.pooling_shape[1]] += error_tensor[b,c,yy,xx] * mask_window
                        xx += 1
                    yy += 1

        return output
