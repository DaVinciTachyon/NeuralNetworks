from numpy import *
import numpy as np
from bigfloat import *
import bigfloat as bf

class NeuralNetwork:
    def __init__(self, inputs, y): #number of layers, array of nodes per layer, add biases
        self.input      = inputs
        self.y          = y
        self.weights = [np.random.rand(self.input.shape[1], 3), np.random.rand(3, 2), np.random.rand(2, 2)]
        bf.exp(5000,bf.precision(100))
        self.biases = [np.random.rand(3), np.random.rand(2), np.random.rand(2)]
        #self.layers = num of layers + 1 for input

    #def addlayer(self, num_of_nodes):

    def feedforward(self, layer):
        for w in self.weights: #+ self.biases using zip?
            layer = sigmoid(np.dot(layer, w))# + b)
        return layer

    def backprop(self):
        layer1 = sigmoid(np.dot(self.input, self.weights[0]))# + self.biases[0])
        layer2 = sigmoid(np.dot(layer1, self.weights[1]))# + self.biases[1])
        output = sigmoid(np.dot(layer2, self.weights[2]))# + self.biases[2])

        w2 = 2 * (self.y - output) * sigmoid_derivative(output)
        d_weights2 = np.dot(layer2.T, w2)
        w1 = np.dot(w2, self.weights[2].T) * sigmoid_derivative(layer2)
        d_weights1 = np.dot(layer1.T, w1)
        w0 = np.dot(w1, self.weights[1].T) * sigmoid_derivative(layer1)
        d_weights0 = np.dot(self.input.T, w0)

        #d_biases2 = 2 * (self.y - output) * sigmoid_derivative(output)
        #d_biases1 = np.dot(d_biases2, self.weights[2].T) * sigmoid_derivative(layer2)
        #d_biases0 = np.dot(d_biases1, self.weights[1].T) * sigmoid_derivative(layer1)

        #self.biases[0] += d_biases0
        #self.biases[1] += d_biases1
        #self.biases[2] += d_biases2

        self.weights[0] += d_weights0
        self.weights[1] += d_weights1
        self.weights[2] += d_weights2

    def train(self, number_of_iterations): #inputs, and outputs for training
        i = 0
        for iteration in xrange(number_of_iterations):
            #self.feedforward(self.input)
            self.backprop()
            i += 1
            if(i % 10000 == 0):
                print(array([1, 0]), neural_network.feedforward(array([0, 0, 0])))
                print(array([0, 1]), neural_network.feedforward(array([1, 1, 1])))


def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

if __name__ == "__main__":
    training_set_inputs = array([[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0]])
    training_set_outputs = array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]])

    neural_network = NeuralNetwork(training_set_inputs, training_set_outputs)

    print(array([1, 0]), neural_network.feedforward(array([0, 0, 0])))
    print(array([0, 1]), neural_network.feedforward(array([1, 1, 1])))

    neural_network.train(1000000)

    print(array([1, 0]), neural_network.feedforward(array([0, 0, 0])))
    print(array([0, 1]), neural_network.feedforward(array([1, 1, 1])))
