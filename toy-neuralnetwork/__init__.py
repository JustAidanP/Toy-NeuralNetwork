from builder import NeuralNetworkBuilder
from model import NeuralNetworkModel, Layer


if __name__ == "__main__":
    import numpy

    # Initialises the xor dataset
    dataset = [
        (Layer.from_list([0, 0]), Layer.from_list([0, 1])),
        (Layer.from_list([0, 1]), Layer.from_list([1, 0])),
        (Layer.from_list([1, 0]), Layer.from_list([1, 0])),
        (Layer.from_list([1, 1]), Layer.from_list([0, 1])),
    ]
    # Builds a neural network with 2 inputs, 2 outputs and a single hidden layer with 4 nodes
    # This network will be initialised with random weights, and biases as well as the Sigmoid activation function
    network = NeuralNetworkBuilder([2, 4, 2]).build()
    # Trains the neural network for 50,000 epochs, each with a batch size of 4
    print("Starting training")
    network.train_model(dataset, 50000, 4, lambda x: 0.1)
    # Prints the results
    print(f"Query(0, 1) -> {network.query(Layer(numpy.array([[0], [0]]))).as_list()}")
    print(f"Query(1, 0) -> {network.query(Layer(numpy.array([[1], [0]]))).as_list()}")
    print(f"Query(0, 1) -> {network.query(Layer(numpy.array([[0], [1]]))).as_list()}")
    print(f"Query(1, 1) -> {network.query(Layer(numpy.array([[1], [1]]))).as_list()}")
