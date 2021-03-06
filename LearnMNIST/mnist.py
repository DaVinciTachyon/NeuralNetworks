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
            neural_network = ConvolutionalNeuralNetwork(row, output, [3, 5, 10])
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

x = 0
while True:
    print x, "-", 1 - neural_network.test()
    neural_network.train(1)
    x = x + 1
    if x > 10:
        break
print "final -", neural_network.test()