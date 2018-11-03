from numpy import *
import numpy as np
from bigfloat import *
import bigfloat as bf

class NeuralNetwork:
    def __init__(self, inputs, y): #number of layers, array of nodes per layer, add biases
        self.input      = inputs
        self.y          = y
        self.number_of_layers = 3
        self.weights = [np.random.rand(self.input.shape[1], 3), np.random.rand(3, 2), np.random.rand(2, self.y.shape[1])]
        bf.exp(5000,bf.precision(100))
        self.biases = [np.random.rand(3), np.random.rand(2), np.random.rand(self.y.shape[1])]
        first = array((self.input.shape[0], self.weights[0].shape[1]))
        second = array((first.shape[0], self.weights[1].shape[1]))
        third = array((second.shape[0], self.weights[2].shape[1]))
        self.layers = [first, second, third]

    #def addlayer(self, num_of_nodes):

    def feedforward(self, layer):
        i = 0
        for w, b in zip(self.weights, self.biases):
            if(i > 0):
                layer = self.layers[i - 1]
            self.layers[i] = sigmoid(np.dot(layer, w) + b)
            i += 1
        return self.layers[self.number_of_layers - 1]

    def backprop(self):
        w2 = 2 * (self.y - self.layers[2]) * sigmoid_derivative(self.layers[2])
        d_weights2 = np.dot(self.layers[1].T, w2)
        w1 = np.dot(w2, self.weights[2].T) * sigmoid_derivative(self.layers[1])
        d_weights1 = np.dot(self.layers[0].T, w1)
        w0 = np.dot(w1, self.weights[1].T) * sigmoid_derivative(self.layers[0])
        d_weights0 = np.dot(self.input.T, w0)

        d_bias = 2 * (self.y - self.layers[2]) * sigmoid_derivative(self.layers[2])
        for x in range(self.number_of_layers):
            self.biases[self.number_of_layers - 1 - x] += d_bias[0]
            if(x < self.number_of_layers - 1):
                d_bias = np.dot(d_bias, self.weights[self.number_of_layers - 1 - x].T) * sigmoid_derivative(self.layers[self.number_of_layers - 2 - x])

        self.weights[0] += d_weights0
        self.weights[1] += d_weights1
        self.weights[2] += d_weights2

    def train(self, number_of_iterations): #inputs, and outputs for training
        for iteration in xrange(number_of_iterations):
            self.feedforward(self.input)
            self.backprop()


def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

if __name__ == "__main__":
    training_set_inputs = array([[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0]])
    training_set_outputs = array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]])

    neural_network = NeuralNetwork(training_set_inputs, training_set_outputs)

    for x in range(100):
        print "desired: ", array([1, 0]), " - actual: ", neural_network.feedforward(array([0, 0, 0]))
        print "desired: ", array([0, 1]), " - actual: ", neural_network.feedforward(array([1, 1, 1]))
        print

        neural_network.train(10000)
