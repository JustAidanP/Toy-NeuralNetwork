from .protocol import ActivationFunction


class Relu(ActivationFunction):
    """
    Used for AutoEncoders
    Takes in a value and returns it if it is above 0
    """

    # noinspection PyMethodMayBeStatic
    def at(self, x: float) -> float:
        if x > 0:
            return x
        else:
            return 0

    # noinspection PyMethodMayBeStatic
    def derivative_at(self, x: float) -> float:
        if x < 0:
            return 0
        else:
            return 1
