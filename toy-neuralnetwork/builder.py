import numpy

from activation_functions import ActivationFunction, Sigmoid
from model import NeuralNetworkModel


class NeuralNetworkBuilder:
    node_counts: list[int]
    # Defaults to the sigmoid function
    activation_functions: list[ActivationFunction]
    # Defaults to uniformly random values between -1 and 1
    weights: list[numpy.ndarray]
    # Defaults to uniformly random values between -1 and 1
    biases: list[numpy.ndarray]

    def __init__(self, node_counts: list[int]):
        self.node_counts = node_counts
        self.with_single_activation_function(Sigmoid())
        self.with_random_weights()
        self.with_random_biases()

    def with_random_weights(self):
        """
        Creates an initial (random) set of weights between each layer, normalised to between -1 and 1,
        mathematically speaking, the weights between layer i and layer i + 1, are a matrix of size |i+1| x |i|,
        where element n,m of the matrix, is the weight between node m in layer i and node n in layer i + 1.

        :return:
        """
        self.weights = [
            (numpy.random.rand(self.node_counts[i + 1], self.node_counts[i]) * 2 - 1)
            / self.node_counts[i]
            for i in range(len(self.node_counts) - 1)
        ]

    def with_random_biases(self):
        """
        Creates an initial (random) set of biases for each (non-input) layer (each node in each layer gets its on bias),
        normalised to between -1 and 1. The biases are stored as a column matrix (a matrix of size nx1 where n is the
        size of the corresponding layer)

        :return:
        """
        self.biases = [
            (numpy.random.rand(self.node_counts[i + 1], 1) * 2 - 1)
            / self.node_counts[i]
            for i in range(len(self.node_counts) - 1)
        ]

    def with_activation_functions(self, activation_functions: list[ActivationFunction]):
        """
        All but the initial layer require an activation function, this assigns the provided activation functions to
        their respective layers, activation function with index i gets assigned to layer i + 1 (as such, there must
        n - 1 activation functions provided where n is the number of layers)

        :param activation_functions: The functions to assign to their layers
        """
        if len(activation_functions) != len(self.node_counts) - 1:
            raise ValueError(
                "Please provide the correct number of activation functions"
            )
        self.activation_functions = activation_functions

    def with_single_activation_function(self, activation_function: ActivationFunction):
        """
        All but the initial layer require an activation function, this assigns the provided activation function to all
        layers

        :param activation_function: The activation function to assign to every layer
        """
        self.activation_functions = [
            activation_function for _ in range(len(self.node_counts) - 1)
        ]

    def build(self) -> NeuralNetworkModel:
        """
        Builds the neural network model, using the parameters provided

        :return: The model
        """
        return NeuralNetworkModel(
            self.node_counts, self.activation_functions, self.weights, self.biases
        )
