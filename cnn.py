from numpy import dot, exp, array
import numpy as np
from bigfloat import exp, precision
import bigfloat as bf
#store training data, have the ability to add data or clear all data
#get percentage error
#train by giving or not giving data
#train until a specific margin of error

class ConvolutionalNeuralNetwork:
    #sample input, sample output, array of number of nodes in hidden layer
    def __init__(self, input, output, nodes = [3, 2]):
        bf.exp(5000,bf.precision(100))
        nodes_in_layer = nodes
        nodes_in_layer.append(len(output))
        self.number_of_layers = len(nodes_in_layer)
        self.weights = []
        self.weights.append(np.random.rand(len(input), nodes_in_layer[0]))
        for x in range(self.number_of_layers - 1):
            self.weights.append(np.random.rand(nodes_in_layer[x], nodes_in_layer[x + 1]))
        self.biases = []
        for x in range(self.number_of_layers):
            self.biases.append(np.random.rand(nodes_in_layer[x]))
        self.layers = []
        prev = array((len(input), self.weights[0].shape[1]))
        for x in range(self.number_of_layers):
            self.layers.append(prev)
            if(x < self.number_of_layers - 1):
                prev = array((prev.shape[0], self.weights[x + 1].shape[1]))

    def feedforward(self, layer):
        i = 0
        for w, b in zip(self.weights, self.biases):
            if(i > 0):
                layer = self.layers[i - 1]
            self.layers[i] = sigmoid(np.dot(layer, w) + b)
            i += 1
        return self.layers[self.number_of_layers - 1]

    def backprop(self):
        d_weights = []
        for x in range(len(self.weights)):
            d_weights.append(array((self.weights[x].shape[0], self.weights[x].shape[1])))

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

    def train(self, number_of_iterations, inputs, outputs):
        self.input = inputs
        self.y = outputs
        for iteration in xrange(number_of_iterations):
            self.feedforward(self.input)
            self.backprop()

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)
