from numpy import *
import numpy as np

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],4)
        self.weights2   = np.random.rand(4,1)
        self.y          = y
        self.output     = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def train(self, number_of_iterations):
        for iteration in xrange(number_of_iterations):
            self.feedforward()
            self.backprop()

    def think(self, input):
        layer1 = sigmoid(np.dot(input, self.weights1))
        return sigmoid(np.dot(layer1, self.weights2))

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

if __name__ == "__main__":
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    neural_network = NeuralNetwork(training_set_inputs, training_set_outputs)

    print("0", neural_network.think(array([0, 0, 0])))
    print("1", neural_network.think(array([1, 0, 0])))

    neural_network.train(10000)

    print("0", neural_network.think(array([0, 0, 0])))
    print("1", neural_network.think(array([1, 0, 0])))
