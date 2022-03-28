# Copyright © 2018 JustAidanP. All rights reserved.
from typing import List

import numpy, time, random, sys


class ActivationFunctions:
    """
    The available activation functions
    """
    # ------Activation Functions------
    @staticmethod
    def sigmoid(x, derivative=False):
        """
        Sigmoid is used for getting an output as a probability of most likely, i.e. classifiers
        The sigmoid function takes in a value and normalises to between 0 and 1 with the sigmoid function,
        can also return the derivative
        Takes in a float and boolean and returns a float

        :param x: The input to the function
        :param derivative: Whether the derivative of the input should be returned
        :return:
        """
        if derivative:
            return x * (1.0 - x)
        return 1 / (1 + numpy.exp(-x))

    @staticmethod
    def tanh(x, derivative=False):
        """
        TanH is used for getting an output as a probability of most likely i.e. classifiers
        The TanH functions takes in a value and normalises to between -1 and 1, can also return the derivative
        Takes in a float and returns the float

        :param x: The input to the function
        :param derivative: Whether the derivative oof the input should be returned
        :return:
        """
        if derivative:
            return 1 - (ActivationFunctions.tanh(x) ** 2)
        return 2 / (1 + numpy.exp(-2 * x)) - 1

    @staticmethod
    def relu(x, derivative=False):
        """
        Used for AutoEncoders
        Takes in a value and returns it if it is above 0

        :param x: The input to the function
        :param derivative: Whether the derivative oof the input should be returned
        :return:
        """
        if derivative:
            if x < 0: return 0
            return 1
        if x > 0: return x
        return 0

    @staticmethod
    def leaky_relu(x, derivative=False):
        """
        Used for AutoEncoders
        Takes in a value and returns it if it is above 0, if not it returns a linear function of it

        :param x: The input to the function
        :param derivative: Whether the derivative oof the input should be returned
        :return:
        """
        if derivative:
            if x < 0: return 0.01
            return 1
        if x > 0: return x
        return 0.01 * x

    @staticmethod
    def soft_plus(x, derivative=False):
        """
        Used for AutoEncoders
        Takes in a value and returns the softplus function of it

        :param x: The input to the function
        :param derivative: Whether the derivative oof the input should be returned
        :return:
        """
        if derivative:
            return 1 / (1 + numpy.exp(-x))
        return numpy.log(1 + numpy.exp(x))

    # ------Vectorized Functions------
    Sigmoid = numpy.vectorize(sigmoid)
    TanH = numpy.vectorize(tanh)
    Relu = numpy.vectorize(relu)
    LeakyRelu = numpy.vectorize(leaky_relu)
    SoftPlus = numpy.vectorize(soft_plus)


class NeuralNetwork:
    """
    On initialisation the Neural Network class will create a set of weights for use
    A Neural Network module for machine learning
    """
    # ------Varibales------
    # Defines the rate at which the weights are adjusted
    LEARNINGRATE = 0.01
    # Stores the initial error
    initial_error = None

    def __init__(self, node_counts: List[int], activations=[ActivationFunctions.Sigmoid]):
        """
        Initialises the neural network

        :param node_counts: The number of nodes in each layer of the neural network. There must be at least 2 elements,
        With an input of n elements, element 0, is the size of the input layer;
        element i, 1 <= i < n - 1, is the of hidden layer i - 1 (layer i);
        and element n-1, is the size of the output layer.

        :param activations: The activation function to apply to each layer. Either a single activation function can be
        given, which would then be applied to every layer; or n - 1 (as defined in `node_counts`) functions can be
        given, where function i, would be applied to layer i + 1.
        """
        # ------Assigns activation functions------
        # Assigns the chosen activation function to all if only one was given
        if len(activations) == 1:
            self.activation_functions = [activations[0] for i in range(len(node_counts) - 1)]
        elif len(activations) == len(node_counts) - 1:
            self.activation_functions = activations
        else:
            print("Activations not of length %s, Can't Run Neural Network" % (len(node_counts) - 1));
            sys.exit(0)
        # ------Sets up instance properties------
        # Layers stores a list of matrices, each matrix storing the nodes for the layer
        self.node_layers = []
        # Creates a list for every layer to layer weights
        # Each index item is a matrix for the layer to layer weights
        self.weights = []
        # Creates a list for every activated layer's biases
        # The bias will be added on after the layer has been calculated
        self.biases = []
        # Stores the gradients and weight deltas of a batch
        self.batch_gradients = []
        self.batch_weight_deltas = []
        # ------Assigning random weights------
        # Assigns the layer to layer weights, in a way that the weights connecting to a node will sum up to max 1 initialy
        # The sizes of the matrix is the size of the next layer by the previous layer to create the correct sized weight matrix
        # An array is used for a performance gain, the array is used in the same way as a matrix
        self.weights = [
            numpy.array(numpy.random.rand(node_counts[layerIter + 1], node_counts[layerIter]) * 2 - 1) / node_counts[
                layerIter] for layerIter in range(len(node_counts) - 1)]
        # self.weights =  [numpy.array(numpy.full((nodeCounts[layerIter+1], nodeCounts[layerIter]), 1)) / nodeCounts[layerIter] for layerIter in range(len(nodeCounts) - 1)]
        # ------Assigning random biases------
        # Assigns the biases to be added onto every activated layey, each one between -1 and 1
        # The size of the matrix is the size of the layer that the biases will be applied to
        # An array is used for a performance gain, the array is used in the same way as a matrix
        self.biases = [numpy.array(numpy.random.rand(node_counts[layerIter + 1]) * 2 - 1) / node_counts[layerIter] for
                       layerIter in range(len(node_counts) - 1)]

    # ------Methods/ Functions------
    def propagate_network(self, layer_range):
        """
        Propagates the neural network forward, finds the dot product of the weights by the previous layer
        and applies the activation function to each element
        takes a range of layers to propagate

        :param layer_range: The range of layers to evaluate
        :return:
        """
        # Creates the nodes for every layer in the network
        # Iterated based on the count of weights as the layers aren't defined as of this point
        for layerIter in layer_range:
            # Calculates the matrix multiplication of the weight matrix connecting the previous layer to the current
            # by the previous nodeLayer to get a list of unactivated values for the current layer
            dot_products_of_layer = (self.weights[layerIter] @ self.node_layers[layerIter])
            # Adds the biases to the new layer
            dot_products_of_layer = dot_products_of_layer + self.biases[layerIter]
            # Applies the selected activation function to every elemnt in the new matrix and appends it to nodeLayers
            self.node_layers.append(self.activation_functions[layerIter](dot_products_of_layer))
            # self.nodeLayers.append(dot_products_of_layer)

    def query(self, layer_nodes, layer_range=None):
        """
        Queries the neural network based on a given layer and a range of layers
        The layer must be an array and will be converted to a matrix array
        Returns the result of the query

        :param layer_nodes:
        :param layer_range:
        :return:
        """
        if layer_range is None:
            layer_range = range(len(self.weights))
        # Resets the nodeLayers variable for a successful query, filling in layers that aren't needed
        self.node_layers = [numpy.array([]) for i in range(layer_range[0])]
        # Adds the layerNodes to the layers to be executed, has to be transposed
        self.node_layers.append(numpy.array(layer_nodes).T)
        # Propagates the network to get an output
        self.propagate_network(layer_range)
        # Returns the final output layer
        return self.node_layers[-1]

    # ------Backpropagation------
    def sgd(self, desired_output):
        """
        Performs stochastic gradient descent and returns the results
        :param desired_output:
        :return:
        """
        # ------Calculates the errors of every layer------
        # Creates a list that contains the errors of the final layer
        # Performs the error function on the differenct between the desiredOutput and the final node_layer
        errors = [
            numpy.array(
                [-(error ** 2) if error < 0 else error ** 2 for error in desired_output - self.node_layers[-1]])]

        # Calculates the error of every layer, not including the first of last
        # Calculates the errors in reverse order
        for error_iter in range(len(self.node_layers) - 2, 0, -1):
            # Calculates the error of the layer using the next layers error
            # And the weights between the layer and the next
            errors.insert(0, numpy.dot(errors[0], self.weights[error_iter]))

        # Stores a list of gradients and weight_deltas for each layer
        gradients = []
        weight_deltas = []
        # Adjusts every weight to minimise the error using gradient descent
        for layer_iter in range(len(self.weights)):
            # Creates a transposed copy of the current layer
            node_layer = numpy.array(self.node_layers[layer_iter + 1])
            # Calculates the gradient of the error to weight relationship using the derivative of the activation function and the errors
            gradients.append(
                numpy.array(errors[layer_iter] * self.activation_functions[layer_iter](node_layer, derivative=True),
                            ndmin=2).T)
            # Calculates the deltas for the weights
            # Multiplies the dot product of the gradent and previous layer by the learning rate
            weight_deltas.append(
                NeuralNetwork.LEARNINGRATE * (gradients[-1] @ numpy.array(self.node_layers[layer_iter].T, ndmin=2)))
        return numpy.array(gradients), numpy.array(weight_deltas)

    def adjust_from_batch(self, gradients, weight_deltas):
        """
        Adjusts the network based on the last batch

        :param gradients:
        :param weight_deltas:
        :return:
        """
        for layer_iter in range(len(gradients)):
            self.weights[layer_iter] += weight_deltas[layer_iter]
            # Changes the biases based on the gradients, transposes the gradient and converts it to a 1-dimensional array for addition
            self.biases[layer_iter] += NeuralNetwork.LEARNINGRATE * gradients[layer_iter].T[0]
            # Normalises the biases to between -1 and 1
            if numpy.amax(self.biases[layer_iter]) > 1: self.biases[layer_iter] = self.biases[layer_iter] / numpy.amax(
                self.biases[layer_iter])

    def back_propogate(self, desired_output):
        """
        Trains all of the weights and biases in the network by use of gradient descent
        Takes in a list of what the final layer should be for calculating the error

        :param desired_output:
        :return:
        """
        # ------Calculates the errors of every layer------
        # Creates a list that contains the errors of the final layer
        # Performs the error function on the difference between the desiredOutput and the final node_layer
        errors = [
            numpy.array(
                [-(error ** 2) if error < 0 else error ** 2 for error in desired_output - self.node_layers[-1]])]

        # Calculates the error of every layer, not including the first of last
        # Calculates the errors in reverse order
        for error_iter in range(len(self.node_layers) - 2, 0, -1):
            # Calculates the error of the layer using the next layers error
            # And the weights between the layer and the next
            errors.insert(0, numpy.dot(errors[0], self.weights[error_iter]))
        # Adjusts every weight to minimise the error using gradient descent
        for layer_iter in range(len(self.weights)):
            # Creates a transposed copy of the current layer
            node_layer = numpy.array(self.node_layers[layer_iter + 1])
            # Calculates the gradient of the error to weight relationship using the derivative of the activation function and the errors
            gradient = numpy.array(
                errors[layer_iter] * self.activation_functions[layer_iter](node_layer, derivative=True),
                ndmin=2).T
            # Calculates the deltas for the weights
            # Multiplies the dot product of the gradent and previous layer by the learning rate
            weight_deltas = NeuralNetwork.LEARNINGRATE * (
                        gradient @ numpy.array(self.node_layers[layer_iter].T, ndmin=2))
            # Adds the weight_deltas onto the weights between the layers
            self.weights[layer_iter] += weight_deltas
            # Changes the biases based on the gradients, transposes the gradient and converts it to a 1-dimensional array for addition
            self.biases[layer_iter] += NeuralNetwork.LEARNINGRATE * gradient.T[0]
            # Normalises the biases to between -1 and 1
            if numpy.amax(self.biases[layer_iter]) > 1: self.biases[layer_iter] = self.biases[layer_iter] / numpy.amax(
                self.biases[layer_iter])

    def train(self, input_data, desired_output_data, epochs, batch_size, lr_min=0.01, lr_max=0.1):
        """
        Called to train the network based on a given traning dataset
        Takes in a minimum learning rate and a maximum learning rate

        :param input_data:
        :param desired_output_data:
        :param epochs:
        :param batch_size:
        :param lr_min:
        :param lr_max:
        :return:
        """
        # Keeps track of the last 200 that were right
        # last200 = []

        # Stores the gradients and wightDeltas of the batch
        batch_gradients = numpy.array([])
        batch_weight_deltas = numpy.array([])
        # Stores the current iteration through the batch
        batch_iteration = 0

        # Loops through the training data based on noOfLoops
        for j in range(epochs):
            # Duplicates the inputData as a backup when the input data is deleted during runtime
            input_ref = [i for i in range(len(input_data))]
            # Duplicates the desiredOutputData as a backup when the desiredOutput data is deleted during runtime
            desired_output_ref = [i for i in range(len(desired_output_data))]

            NeuralNetwork.LEARNINGRATE = lr_max - ((lr_max - lr_min) * j / epochs)

            # Runs through every piece of training data
            for i in range(len(input_data)):
                # Randomly chooses an index of the training data to be used
                chosen_train_index = numpy.random.randint(0, len(input_ref))
                chosen_train_iter = input_ref[chosen_train_index]
                # Queries the network with the randomly chosen input data
                output = self.query(input_data[chosen_train_iter])
                # Runs stochastic gradient descent
                gradients, weight_deltas = self.sgd(desired_output_data[chosen_train_iter])
                # Adds the results to the batch record
                if len(batch_gradients) == 0:
                    batch_gradients = gradients
                else:
                    batch_gradients += gradients
                if len(batch_weight_deltas) == 0:
                    batch_weight_deltas = weight_deltas
                else:
                    batch_weight_deltas += weight_deltas

                if batch_iteration == batch_size:
                    # Gets the average of the batch
                    for gradient in batch_gradients: gradient /= batch_size
                    for weight_deltas in batch_weight_deltas: weight_deltas /= batch_size
                    # Adjusts the network based on the batch
                    self.adjust_from_batch(batch_gradients, batch_weight_deltas)
                    # Resets batch variables
                    batch_gradients = numpy.array([])
                    batch_weight_deltas = numpy.array([])
                    batch_iteration = 0
                batch_iteration += 1

                # #Backpropgates the network with the correct output for the random input data
                # self.backPropogate(desiredOutputData[chosen_train_iter])

                # Prints a progress every 1000 iterations
                if ((j * len(input_data)) + i) % 1000 == 0: print(str(((j * len(input_data)) + i) // 1000) + "--")

                # Removes the training data used during this loop of training
                input_ref.pop(chosen_train_index)
                desired_output_ref.pop(chosen_train_index)

    def export_weights_and_biases(self, destination):
        """
        Saves the weights and biases to a file

        :param destination:
        :return:
        """
        numpy.save(destination + "weights", numpy.array(self.weights))
        numpy.save(destination + "biases", numpy.array(self.biases))

    def load_weights_and_biases(self, location):
        """
        Loads the weights and biases from a file

        :param location:
        :return:
        """
        self.weights = [element for element in numpy.load(location + "weights.npy")]
        self.biases = [element for element in numpy.load(location + "biases.npy")]


if __name__ == "__main__":
    numpy.seterr("raise")
    neural_net = NeuralNetwork(node_counts=[2, 4, 2], activations=[ActivationFunctions.SoftPlus])
    # print(neuralNet.weights[0])
    input = [[0, 0], [0, 1], [1, 0], [1, 1]]
    output = [[0, 1], [1, 0], [1, 0], [0, 1]]
    neural_net.train(input_data=input, desired_output_data=output, epochs=10000, batch_size=4)
    # neuralNet.loadWeighctsAndBiases("/Users/NAME/Desktop/Machine Learning/")
    print(neural_net.query(input[0]))
    print(neural_net.query(input[1]))
    print(neural_net.query(input[2]))
    print(neural_net.query(input[3]))
    # print(neuralNet.nodeLayers)
    # startTime = time.time()
    # for i in range(100):
    #     neuralNet.query([1, 1], getBest=False)
    # endTime = time.time()
    # print("Time: %s"%(endTime-startTime))
