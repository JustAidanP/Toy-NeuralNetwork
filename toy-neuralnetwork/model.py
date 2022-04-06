import random
import typing

import numpy
from activation_functions import ActivationFunction


class Layer:
    def __init__(self, inner_matrix: numpy.ndarray):
        if inner_matrix.ndim != 2 or inner_matrix.shape[1] != 1:
            raise ValueError("The matrix should have dimensions nx1 for some n")
        self.inner_matrix = inner_matrix

    def get_inner_matrix(self) -> numpy.ndarray:
        return self.inner_matrix

    @classmethod
    def from_list(cls, nodes: list[float]):
        return Layer(numpy.array(nodes, ndmin=2).T)

    def as_list(self) -> list[float]:
        return list(self.inner_matrix.flatten())


class NeuralNetworkModel:
    """
    1. Each layer is a column matrix, i.e. a transposed vector
    2. The weights get applied as W . L_i = L_(i+1) where W is the weight matrix between layer i and i + 1
    """

    node_counts: list[int]
    activation_functions: list[ActivationFunction]
    weights: list[numpy.ndarray]
    biases: list[numpy.ndarray]

    def __init__(
        self,
        node_counts: list[int],
        activation_functions: list[ActivationFunction],
        weights: list[numpy.ndarray],
        biases: list[numpy.ndarray],
    ):
        self.node_counts = node_counts
        # Activation function validation [excludes type validation]
        if len(activation_functions) != len(self.node_counts) - 1:
            raise ValueError(
                "The number of activation functions must be n - 1 where n is the number of layers of the model"
            )
        else:
            self.activation_functions = activation_functions
        # Ensures that the correct number of weight matrices are provided
        if len(weights) != len(self.node_counts) - 1:
            raise ValueError(
                "The number of weight matrices provided must be n - 1 where n is the number of layers of the model"
            )
        # Ensures that each weight matrix has correct size
        for i, weight_matrix in enumerate(weights):
            if weight_matrix.shape != (self.node_counts[i + 1], self.node_counts[i]):
                raise ValueError(
                    "Ensure that all weight matrices are of the correct shape"
                )
        self.weights = weights
        # Ensures that the correct number of bias vectors are provided
        if len(biases) != len(self.node_counts) - 1:
            raise ValueError(
                "The number of bias vectors provided must be n - 1 where n is the number of layers of the model"
            )
        # Ensure that each bias vector is of the correct shape (nx1)
        for i, bias in enumerate(biases):
            if bias.shape != (self.node_counts[i + 1], 1):
                raise ValueError("Ensure that each bias matrix is of the correct shape")
        self.biases = biases

    def query(self, input_layer: Layer) -> Layer:
        """
        Queries the full neural network, with the inputs given
        :return: The output layer
        """
        return self.query_subrange(input_layer, 0, len(self.node_counts) - 1)

    def query_subrange(
        self, starting_layer: Layer, starting_from: int, upto: int
    ) -> Layer:
        """
        Queries a subrange of the neural network with `starting_from` being the index (starting at 0) of the input layer
        and `upto` being the index of the layer to output (i.e. the layer queried to).

        For example, if there are 5 layers,
            1. With `starting_from` = 0, and `upto` = 4, the model's input layer is the starting layer and the model's
            output layer is what gets output
            2. With `starting_from` = 2, and `upto` = 3, the input provided is for the model's third layer and the
            model's 4th layer is what gets output
        """
        # Ensures that the starting_layer is of the correct size
        if starting_layer.get_inner_matrix().shape != (
            self.node_counts[starting_from],
            1,
        ):
            raise ValueError("Ensure that the input_layer is of the correct size")
        # Propagates the neural network
        current_layer = starting_layer.get_inner_matrix()
        for i in range(starting_from, upto):
            weights = self.weights[i]
            biases = self.biases[i]
            activation_function = self.activation_functions[i]
            current_layer = numpy.array(
                [
                    [activation_function.at(x)]
                    for [x] in biases + weights @ current_layer
                ]
            )
        return Layer(current_layer)

    def __back_propagate(
        self, datapoint: typing.Tuple[Layer, Layer]
    ) -> typing.Tuple[list[numpy.ndarray], list[numpy.ndarray]]:
        """
        Calculates the changes required to the weights and biases to make this model fit better to this data point using
        back propagation

        :param datapoint: The data point to perform back propagation on
        :return: (Weight Deltas, Bias Deltas) With same indexing, sizes, etc. as `self.weights` and
                `self.biases` respectively
        """
        input_data, output_data = datapoint
        # First, queries the model for the datapoint, it does this by querying between each layer
        query_results = [input_data]
        for i in range(len(self.node_counts) - 1):
            query_results.append(self.query_subrange(query_results[-1], i, i + 1))

        # Calculates the square errors (maintaining sign) for each layer, this is achieved by first calculating the
        # distance between the output datapoint and the output layer from the query, and then propagating this backwards
        errors = [
            numpy.array(
                [
                    [-(error**2) if error < 0 else error**2]
                    for [error] in output_data.get_inner_matrix()
                    - query_results[-1].get_inner_matrix()
                ]
            )
        ]

        # Propagates this error backwards, we reverse through the layers, excluding both the input and output layer,
        # propagating the error as we go [i.e. element i relates to layer i + 1]
        for i in range(len(self.node_counts) - 2, 0, -1):
            errors.append(self.weights[i].T @ errors[-1])
        # Reverses the errors list so that the errors match up with their respective layers
        errors.reverse()

        # Stores all calculated deltas
        bias_deltas = []
        weight_deltas = []
        for i in range(1, len(self.node_counts)):
            # We are operating with respect to layer i (with errors := errors[i-1]), and the weights connecting into
            # said layer
            layer_errors = errors[i - 1]
            layer = query_results[i]
            activation_function = self.activation_functions[i - 1]
            # Calculates the gradient which we want to apply to
            #   (a) The biases, and crucially
            #   (b) The weights
            # Via element-wise multiplication between the derivative of the activation of each node (in the layer),
            # and it's respective error
            gradients = numpy.array(
                [
                    [e * activation_function.derivative_at(x)]
                    for ([x], [e]) in zip(layer.get_inner_matrix(), layer_errors)
                ]
            )
            bias_deltas.append(gradients)
            # Uses the gradients to calculate the deltas that we want to change the weights by, this is calculated
            # using the layer that the weights connect from
            weight_deltas.append(gradients @ query_results[i - 1].get_inner_matrix().T)

        return weight_deltas, bias_deltas

    def train_model(
        self,
        dataset: list[typing.Tuple[Layer, Layer]],
        epochs: int,
        batch_size: int,
        learning_rate_function: typing.Callable[[int], float],
    ):
        """
        Trains the network using stochastic gradient descent. i.e. in batches.
        For every batch (epoch), a random selection of `batch_size` data points are chosen from the dataset,
        queried for errors in the current model, errors are then accumulated, and the weights and biases are
        adjusted according to the errors

        :param dataset: The dataset to train on, expressed as a set of inputs matched to their outputs
        :param epochs: The number of batches to run
        :param batch_size: The size of each batch
        :param learning_rate_function: A function which given the current epoch, provides the learning rate
        :return:
        """
        for i in range(epochs):
            learning_rate = learning_rate_function(i)
            # Iterates through a uniformly random selection of `batch_size` data points
            cumulative_weight_deltas = None
            cumulative_bias_deltas = None
            for datapoint in random.sample(dataset, batch_size):
                # Performs stochastic gradient descent on this datapoint to get the gradients
                weight_deltas, bias_deltas = self.__back_propagate(datapoint)
                # Adds the deltas to the cumulative total
                if cumulative_weight_deltas is None:
                    cumulative_weight_deltas = weight_deltas
                else:
                    cumulative_weight_deltas = [
                        cumulative + new
                        for (cumulative, new) in zip(
                            cumulative_weight_deltas, weight_deltas
                        )
                    ]
                # Likewise for biases
                if cumulative_bias_deltas is None:
                    cumulative_bias_deltas = bias_deltas
                else:
                    cumulative_bias_deltas = [
                        cumulative + new
                        for (cumulative, new) in zip(
                            cumulative_bias_deltas, bias_deltas
                        )
                    ]
            # Adjusts the models weights and biases according to the normalised cumulative changes
            self.weights = [
                weights + learning_rate * cumulative / batch_size
                for weights, cumulative in zip(self.weights, cumulative_weight_deltas)
            ]
            self.biases = [
                biases + learning_rate * cumulative / batch_size
                for biases, cumulative in zip(self.biases, cumulative_bias_deltas)
            ]
