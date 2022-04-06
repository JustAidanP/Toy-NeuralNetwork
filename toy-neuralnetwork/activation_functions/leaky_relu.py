from .protocol import ActivationFunction


class LeakyRelu(ActivationFunction):
    """
    Used for AutoEncoders
    Takes in a value and returns it if it is above 0, if not it returns a linear function of it
    """

    # noinspection PyMethodMayBeStatic
    def at(self, x: float) -> float:
        if x > 0:
            return x
        else:
            return 0.01 * x

    # noinspection PyMethodMayBeStatic
    def derivative_at(self, x: float) -> float:
        if x < 0:
            return 0.01
        else:
            return 1
