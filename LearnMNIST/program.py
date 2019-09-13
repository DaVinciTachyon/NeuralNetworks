import cnn_lague as nn
import numpy as np

with np.load('mnist.npz') as data:
    training_images = data['training_images']
    training_labels = data['training_labels']
    test_images = data['test_images']
    test_labels = data['test_labels']
    validation_images = data['validation_images']
    validation_labels = data['validation_labels']

'''
import matplotlib.pyplot as plot
plot.imshow(training_images[0].reshape(28, 28), cmap = 'gray')
plot.show()
'''

layer_sizes = (784, 20, 10)

net = nn.NeuralNetwork(layer_sizes)
print(net.accuracy(test_images, test_labels))