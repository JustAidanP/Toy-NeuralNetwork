#Copyright © 2018 JustAidanP. All rights reserved.
import numpy, time, random, sys, Debug
#An enumerator for every activation function
class ActivationFunctions:
    iter = 0
    #------Activation Functions------
    #Sigmoid is used for getting an output as a probability of most likely, i.e. classifiers
    #The sigmoid function takes in a value and normalises to between 0 and 1 with the sigmoid function, can also return the derivative
    #Takes in a float and boolean and returns a float
    def sigmoid(x, derivative=False):
        if derivative: return x * (1.0 - x)
        return 1/(1+numpy.exp(-x))
    #TanH is used for getting an output as a probability of most likely i.e. classifiers
    #The TanH functions takes in a value and normalises to between -1 and 1, can also return the derivative
    #Takes in a float and returns the float
    def tanH(x, derivative=False):
        if derivative == False: ActivationFunctions.iter += 1
        try:
            if derivative: return 1- (ActivationFunctions.tanH(x) ** 2)
            return 2/(1+numpy.exp(-2*x)) - 1
        except:
            print("--" + str(ActivationFunctions.iter))
            print(x)
            # print(x)
            sys.exit(0)
    #Used for AutoEncoders
    #Takes in a value and returns it if it is above 0
    def relu(x, derivative=False):
        if derivative:
            if x < 0: return 0
            return 1
        if x > 0: return x
        return 0
    #Used for AutoEncoders
    #Takes in a value and returns it if it is above 0, if not it returns a linear function of it
    def leakyRelu(x, derivative=False):
        if derivative:
            if x < 0: return 0.01
            return 1
        if x > 0: return x
        return 0.01 * x
    #Used for AutoEncoders
    #Takes in a value and returns the softplus function of it
    def softPlus(x, derivative=False):
        ActivationFunctions.iter += 1
        try:
            if derivative: return 1/(1+numpy.exp(-x))
            return numpy.log(1+numpy.exp(x))
        except:
            print("Overflow")
            print(derivative)
            print(neuralNet.biases[-1][ActivationFunctions.iter - 1])
            print(numpy.sum(neuralNet.nodeLayers[-1]))
            print(numpy.sum(neuralNet.weights[0][ActivationFunctions.iter - 1]))
            print(numpy.sum(neuralNet.weights[1][ActivationFunctions.iter - 1]))
            print(numpy.sum(neuralNet.weights[2][ActivationFunctions.iter - 1]))
            print(numpy.sum(neuralNet.weights[3][ActivationFunctions.iter - 1]))
            print(x)
            sys.exit()

    #------Vectorized Functions------
    Sigmoid = numpy.vectorize(sigmoid)
    TanH = numpy.vectorize(tanH)
    Relu = numpy.vectorize(relu)
    LeakyRelu = numpy.vectorize(leakyRelu)
    SoftPlus = numpy.vectorize(softPlus)
#On initalisation the Neural Network class will create a set of weights for use
#A Neural Network module for machine learning
class NeuralNetwork:
    #------Varibales------
    #Defines the rate at which the weights are adjusted
    LEARNINGRATE = 0.1

    #------Initialiser------
    #Takes in nodeCounts as an argument, it is a list of the size of every layer, including input and output
    def __init__(self, nodeCounts, activations=[ActivationFunctions.Sigmoid]):
        #------Assigns activation functions------
        #Assigns the chosen activation function to all if only one was given
        if len(activations) == 1: self.activationFunctions = [activations[0] for i in range(len(nodeCounts) - 1)]
        elif len(activations) == len(nodeCounts) - 1: self.activationFunctions = activations
        else: print("Activations not of length %s, Can't Run Neural Network"%(len(nodeCounts) - 1)); sys.exit(0)
        #------Sets up instance properties------
        #Layers stores a list of matrices, each matrix storing the nodes for the layer
        self.nodeLayers = []
        #Creates a list for every layer to layer weights
        #Each index item is a matrix for the layer to layer weights
        self.weights = []
        #Creates a list for every activated layer's biases
        #The bias will be added on after the layer has been calcluated
        self.biases = []
        #------Assigning random weights------
        #Assigns the layer to layer weights, in a way that the weights connecting to a node will sum up to max 1 initialy
        #The sizes of the matrix is the size of the next layer by the previous layer to create the correct sized weight matrix
        #An array is used for a performance gain, the array is used in the same way as a matrix
        self.weights =  [numpy.array(numpy.random.rand(nodeCounts[layerIter+1], nodeCounts[layerIter]) * 2 - 1) / nodeCounts[layerIter] for layerIter in range(len(nodeCounts) - 1)]
        # self.weights =  [numpy.array(numpy.full((nodeCounts[layerIter+1], nodeCounts[layerIter]), 1)) / nodeCounts[layerIter] for layerIter in range(len(nodeCounts) - 1)]
        #------Assigning random biases------
        #Assigns the biases to be added onto every activated layey, each one between -1 and 1
        #The size of the matrix is the size of the layer that the biases will be applied to
        #An array is used for a performance gain, the array is used in the same way as a matrix
        self.biases = [numpy.array(numpy.random.rand(nodeCounts[layerIter+1]) * 2 - 1) for layerIter in range(len(nodeCounts) - 1)]

    #------Methods/ Functions------
    #Propogates the neural network forward, finds the dot product of the weights by the previous layer
    #And applies the activation function to each element
    def propogateNetwork(self):
        #Creates the nodes for every layer in the network
        #Iterated based on the count of weights as the layers aren't defined as of this point
        for layerIter in range(len(self.weights)):
            #Calculates the matrix multiplication of the weight matrix connecting the previous layer to the current
            #by the previous nodeLayer to get a list of unactivated values for the current layer
            dotProductsOfLayer = (self.weights[layerIter] @ self.nodeLayers[layerIter])
            #Adds the biases to the new layer
            dotProductsOfLayer = dotProductsOfLayer + self.biases[layerIter]
            #Applies the selected activation function to every elemnt in the new matrix and appends it to nodeLayers
            self.nodeLayers.append(self.activationFunctions[layerIter](dotProductsOfLayer))
            # self.nodeLayers.append(dotProductsOfLayer)

    #Queries the neural network based on a given input
    #The input must be an array and will be converted to a matrix array
    #Returns the result of the query
    def query(self, inputNodes, getBest=False):
        #Resets the nodeLayers variable for a successful query
        self.nodeLayers = []
        #Sets the first nodeLayer to the input layer as a matrix transposed
        self.nodeLayers.append(numpy.array(inputNodes).T)
        #Propogates the network to get an output
        self.propogateNetwork()
        #Calculates the final layer node with the highest value
        if getBest:
            largestOuput = -1
            largestIndex = 0
            for outputIndex, output in enumerate(self.nodeLayers[-1]):
                if output > largestOuput:
                    largestOuput = output
                    largestIndex = outputIndex
            return largestIndex
        #Returns the final output layer
        return self.nodeLayers[-1]

    #------Backpropogation------
    #Trains all of the weights and biases in the network by use of gradient descent
    #Takes in a list of what the final layer should be for calcualting the error
    def backPropogate(self, desiredOutput):
        #------Calculates the errors of every layer------
        #Creates a list that contains the errors of the final layer
        #Performs the error function on the differenct between the desiredOutput and the final nodeLayer
        errors = [numpy.array([-(error ** 2) if error < 0 else error ** 2 for error in desiredOutput - self.nodeLayers[-1]])]

        #Calculates the error of every layer, not including the first of last
        #Calcualtes the errors in reverse order
        for errorIter in range(len(self.nodeLayers) - 2, 0, -1):
            #Calculates the error of the layer using the next layers error
            #And the weights between the layer and the next
            errors.insert(0, numpy.dot(errors[0], self.weights[errorIter]))
        #Adjusts every weight to minimise the error using gradient descent
        for layerIter in range(len(self.weights)):
            ActivationFunctions.iter = 0
            #Creates a transposed copy of the current layer
            nodeLayer = numpy.array(self.nodeLayers[layerIter+1])
            #Calculates the gradient of the error to weight relationship using the derivative of the activation function and the errors
            # gradient = numpy.array(errors[layerIter] * nodeLayer.T * (1-nodeLayer.T), ndmin=2).T
            # print()
            gradient = numpy.array(errors[layerIter] * self.activationFunctions[layerIter](nodeLayer, derivative=True), ndmin=2).T
            #Calculates the deltas for the weights
            #Multiplies the dot product of the gradent and previous layer by the learning rate
            weightDeltas = NeuralNetwork.LEARNINGRATE * (gradient @ numpy.array(self.nodeLayers[layerIter].T, ndmin=2))
            #Adds the weightDeltas onto the weights between the layers
            self.weights[layerIter] += weightDeltas
            #Changes the biases based on the gradients, transposes the gradient and converts it to a 1-dimensional array for addition
            self.biases[layerIter] += NeuralNetwork.LEARNINGRATE * gradient.T[0]

    #Called to train the network based on a given traning dataset
    def train(self, inputData, desiredOutputData, epochs):
        #Keeps track of the last 200 that were right
        # last200 = []
        #Loops through the training data based on noOfLoops
        for j in range(epochs):
            #Duplicates the inputData as a backup when the input data is deleted during runtime
            inputRef = [inputValue for inputValue in inputData]
            #Duplicates the desiredOutputData as a backup when the desiredOutput data is deleted during runtime
            desiredOutputRef = [desiredOutputValue for desiredOutputValue in desiredOutputData]
            #Runs through every piece of training data
            for i in range(len(inputData)):
                #Randomly chooses an index of the training data to be used
                chosenTrainIter = numpy.random.randint(0, len(inputRef))
                ActivationFunctions.iter = 0
                #Queries the network with the randomly chosen input data
                output = self.query(inputRef[chosenTrainIter], getBest=True)
                #Backpropgates the network with the correct output for the random input data
                self.backPropogate(desiredOutputRef[chosenTrainIter])
                #Prints a progress every 1000 iterations
                if ((j*len(inputData))+i) % 1000 == 0: print(str(((j*len(inputData))+i) // 1000) + "--")
                # if i % 10000 == 0: print(self.nodeLayers[int((len(self.nodeLayers) - 1) / 2)])
                # if output == desiredOutputRef[chosenTrainIter].index(1): last200.append(1)
                # else: last200.append(0)
                # if len(last200) == 201: last200.pop(0)

                #Removes the training data used during this loop of training
                inputRef.pop(chosenTrainIter)
                desiredOutputRef.pop(chosenTrainIter)

    #Saves the weights and biases to a file
    def exportWeightsAndBiases(self, destination):
        numpy.save(destination + "weights", numpy.array(self.weights))
        numpy.save(destination + "biases", numpy.array(self.biases))
    #Loads the weights and biases from a file
    def loadWeightsAndBiases(self, location):
        self.weights = [element for element in numpy.load(location + "weights.npy")]
        self.biases = [element for element in numpy.load(location + "biases.npy")]

if __name__=="__main__":
    global neuralNet

    numpy.seterr("raise")
    neuralNet = NeuralNetwork(nodeCounts=[2, 64, 64, 2], activations=[ActivationFunctions.Sigmoid])
    # print(neuralNet.weights[0])
    # input = [[0,0], [0,1], [1,0], [1,1]]
    # output = [[0,1], [1,0], [1,0], [0,1]]
    # neuralNet.train(inputData=input, desiredOutputData=output, epochs=100000)
    # neuralNet.loadWeightsAndBiases("/Users/aidanprice/Desktop/Machine Learning/")
    # print(neuralNet.query(input[0], getBest=False))
    # print(neuralNet.query(input[1], getBest=False))
    # print(neuralNet.query(input[2], getBest=False))
    # print(neuralNet.query(input[3], getBest=False))
    # print(neuralNet.nodeLayers)
    startTime = time.time()
    for i in range(100):
        neuralNet.query([1, 1], getBest=False)
    endTime = time.time()
    print("Time: %s"%(endTime-startTime))
