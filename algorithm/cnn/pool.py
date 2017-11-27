from .layer import Layer
from tensorflow import nn


class MaxPool(Layer):
    def __init__(self, size=[2, 2], strides=[1, 2, 2, 1], padding="SAME"):
        self.size = size
        self.strides = strides
        self.padding = padding

    def implement(self, x):
        return nn.max_pool(
            x,
            ksize=[1, self.size[0], self.size[1], 1],
            strides=self.strides,
            padding=self.padding)
