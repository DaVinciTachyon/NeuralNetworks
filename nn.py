from numpy import *
import numpy as np
from bigfloat import *
import bigfloat as bf
#store training data, have the ability to add data or clear all data
#automatically test, and give back percentage error
#train by giving or not givin data
#train until a specific margin of error
#take hidden layers as input, but enforce the fact that it is 1 dimensional

class NeuralNetwork:
    def __init__(self, inputs, y):
        self.input      = inputs
        self.y          = y
        nodes_in_layer = [3, 2, self.y.shape[1]]
        self.number_of_layers = len(nodes_in_layer)
        self.weights = [np.random.rand(self.input.shape[1], nodes_in_layer[0]), np.random.rand(nodes_in_layer[0], nodes_in_layer[1]), np.random.rand(nodes_in_layer[1], nodes_in_layer[2])]
        self.biases = [np.random.rand(nodes_in_layer[0]), np.random.rand(nodes_in_layer[1]), np.random.rand(nodes_in_layer[2])]
        first = array((self.input.shape[0], self.weights[0].shape[1]))
        second = array((first.shape[0], self.weights[1].shape[1]))
        third = array((second.shape[0], self.weights[2].shape[1]))
        self.layers = [first, second, third]
        bf.exp(5000,bf.precision(100))

    def feedforward(self, layer):
        i = 0
        for w, b in zip(self.weights, self.biases):
            if(i > 0):
                layer = self.layers[i - 1]
            self.layers[i] = sigmoid(np.dot(layer, w) + b)
            i += 1
        return self.layers[self.number_of_layers - 1]

    def backprop(self):
        d_weights = [array((self.weights[0].shape[0], self.weights[0].shape[1])), array((self.weights[1].shape[0], self.weights[1].shape[1])), array((self.weights[2].shape[0], self.weights[2].shape[1]))]#generalise to any number of layers

        prev = 2 * (self.y - self.layers[self.number_of_layers - 1]) * sigmoid_derivative(self.layers[self.number_of_layers - 1])
        for x in range(self.number_of_layers - 1):
            d_weights[self.number_of_layers - 1 - x] = np.dot(self.layers[self.number_of_layers - 2 - x].T, prev)
            prev = np.dot(prev, self.weights[self.number_of_layers - 1 - x].T) * sigmoid_derivative(self.layers[self.number_of_layers - 2 - x])
        d_weights[0] = np.dot(self.input.T, prev)

        d_bias = 2 * (self.y - self.layers[self.number_of_layers - 1]) * sigmoid_derivative(self.layers[self.number_of_layers - 1])
        for x in range(self.number_of_layers):
            self.biases[self.number_of_layers - 1 - x] += d_bias[0]
            if(x < self.number_of_layers - 1):
                d_bias = np.dot(d_bias, self.weights[self.number_of_layers - 1 - x].T) * sigmoid_derivative(self.layers[self.number_of_layers - 2 - x])

        for x in range(self.number_of_layers):
            self.weights[x] += d_weights[x]

    def train(self, number_of_iterations):
        for iteration in xrange(number_of_iterations):
            self.feedforward(self.input)
            self.backprop()

#make part of class
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

if __name__ == "__main__":
    training_set_inputs = array([[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0]])
    #training_set_outputs = array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]])
    training_set_outputs = array([[0], [0], [0], [1], [1], [1]])

    neural_network = NeuralNetwork(training_set_inputs, training_set_outputs)

    for x in range(10):
        print "desired: ", array([1, 0]), " - actual: ", neural_network.feedforward(array([0, 0, 0]))
        print "desired: ", array([0, 1]), " - actual: ", neural_network.feedforward(array([1, 1, 1]))
        print

        neural_network.train(10000)
