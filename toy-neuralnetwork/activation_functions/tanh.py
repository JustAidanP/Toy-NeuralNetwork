import numpy

from .protocol import ActivationFunction


class Tanh(ActivationFunction):
    """
    Tanh is used for getting an output as a probability of most likely i.e. classifiers
    The Tanh functions takes in a value and normalises to between -1 and 1, can also return the derivative
    Takes in a float and returns the float
    """

    # noinspection PyMethodMayBeStatic
    def at(self, x: float) -> float:
        return 2 / (1 + numpy.exp(-2 * x)) - 1

    # noinspection PyMethodMayBeStatic
    def derivative_at(self, x: float) -> float:
        return 1 - (self.at(x) ** 2)
