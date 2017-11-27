from .layer import Layer
from tensorflow import reduce_mean, nn


class Softmax(Layer):
    def implement(self, x, labels):
        return reduce_mean(
            nn.softmax_cross_entropy_with_logits(labels=labels, logits=x))
