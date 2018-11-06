from numpy import array
from cnn import ConvolutionalNeuralNetwork

training_set_inputs = array([[0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0]])
#training_set_outputs = array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]])
training_set_outputs = array([[0], [0], [0], [1], [1], [1]])

neural_network = ConvolutionalNeuralNetwork(training_set_inputs[0], training_set_outputs[0], [3, 2])

for x in range(10):
    print "desired: ", array([1, 0]), " - actual: ", neural_network.feedforward(array([0, 0, 0]))
    print "desired: ", array([0, 1]), " - actual: ", neural_network.feedforward(array([1, 1, 1]))
    print

    neural_network.train(10000, training_set_inputs, training_set_outputs)
