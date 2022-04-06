import numpy

from .protocol import ActivationFunction


class SoftPlus(ActivationFunction):
    """
    Used for AutoEncoders
    Takes in a value and returns the softplus function of it
    """

    # noinspection PyMethodMayBeStatic
    def at(self, x: float) -> float:
        return numpy.log(1 + numpy.exp(x))

    # noinspection PyMethodMayBeStatic
    def derivative_at(self, x: float) -> float:
        return 1 / (1 + numpy.exp(-x))
