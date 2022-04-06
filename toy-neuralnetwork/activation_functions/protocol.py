import typing


class ActivationFunction(typing.Protocol):
    def at(self, x: float) -> float:
        """
        Returns the value of the function at the given input
        :param x: The input to the function
        :raises ValueError: Raised if the input isn't in the function's domain (function specific)
        :return: The value of the function at the input
        """
        ...

    def derivative_at(self, x: float) -> float:
        """
        Returns the derivative of the function at the given input
        :param x: The input to the function
        :raises ValueError: Raised if the input isn't in the function's domain (function specific)
        :return: The derivative of the function at the input
        """
        ...
