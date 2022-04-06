import numpy

from .protocol import ActivationFunction


class Sigmoid(ActivationFunction):
    """
    Sigmoid is used for getting an output as a probability of most likely, i.e. classifiers
    The sigmoid function takes in a value and normalises to between 0 and 1 with the sigmoid function,
    can also return the derivative
    Takes in a float and boolean and returns a float
    """

    # noinspection PyMethodMayBeStatic
    def at(self, x: float) -> float:
        return 1 / (1 + numpy.exp(-x))

    # noinspection PyMethodMayBeStatic
    def derivative_at(self, x: float) -> float:
        return self.at(x) * (1.0 - self.at(x))
