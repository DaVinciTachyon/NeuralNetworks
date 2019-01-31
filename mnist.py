from numpy import array
from cnn import ConvolutionalNeuralNetwork
import csv

zero    = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
one     = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
two     = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
three   = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
four    = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
five    = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
six     = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
seven   = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
eight   = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
nine    = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

def number(value):
    if(value == '0'):
        return zero
    elif(value == '1'):
        return one
    elif(value == '2'):
        return two
    elif(value == '3'):
        return three
    elif(value == '4'):
        return four
    elif(value == '5'):
        return five
    elif(value == '6'):
        return six
    elif(value == '7'):
        return seven
    elif(value == '8'):
        return eight
    elif(value == '9'):
        return nine

neural_network = None
with open('MNIST/mnist_train.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    print "inputting training data"
    for row in reader:
        output = number(row[0])
        del row[0]
        row = list(map(int, row))
        if(neural_network == None):
            neural_network = ConvolutionalNeuralNetwork(row, output, [30, 20, 16, 12])
        neural_network.addEntry(row, output)
csvFile.close()
with open('MNIST/mnist_test.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    print "inputting test data"
    for row in reader:
        output = number(row[0])
        del row[0]
        row = list(map(int, row))
        neural_network.addTestEntry(row, output)
csvFile.close()

print neural_network.test()
for x in range(100):
    neural_network.train(10000)
    print neural_network.test()