from .layer import Layer
from tensorflow import reshape, matmul
from .variable import *


class FullyConnected(Layer):
    def __init__(self, features):
        self.features = features

    def implement(self, x):
        x_shape = x.get_shape().as_list()
        x_new_shape = 1
        for index, x_i in enumerate(x_shape):
            if index != 0:
                x_new_shape *= x_i
        x_flat = reshape(x, [-1, x_new_shape])
        W = weight([x_new_shape, self.features])
        b = bias([self.features])
        return matmul(x_flat, W) + b