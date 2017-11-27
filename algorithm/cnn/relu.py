from .layer import Layer
from tensorflow import nn


class ReLU(Layer):
    def implement(self, x):
        return nn.relu(x)