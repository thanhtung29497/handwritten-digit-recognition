"""
This is docstring
"""
from tensorflow import truncated_normal, Variable, constant


def weight(shape):
    initial = truncated_normal(shape, stddev=0.1)
    return Variable(initial)


def bias(shape):
    initial = constant(0.1, shape=shape)
    return Variable(initial)