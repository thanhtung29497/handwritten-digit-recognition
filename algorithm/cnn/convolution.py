from .layer import Layer
from .variable import *
from tensorflow import nn


class Convolution2D(Layer):
    def __init__(self, size, features, padding="SAME", strides=[1, 1, 1, 1]):
        self.size = size
        self.features = features
        self.padding = padding
        self.strides = strides

    def implement(self, x):
        W = weight([
            self.size[0], self.size[1],
            x.get_shape().as_list()[3], self.features
        ])
        b = bias([self.features])
        return nn.conv2d(x, W, strides=self.strides, padding=self.padding) + b
