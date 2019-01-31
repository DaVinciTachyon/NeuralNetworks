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

neural_network = None
with open('MNIST/mnist_train.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    print "inputting training data"
    for row in reader:
        output = None
        if(row[0] == '0'):
            output = zero
        elif(row[0] == '1'):
            output = one
        elif(row[0] == '2'):
            output = two
        elif(row[0] == '3'):
            output = three
        elif(row[0] == '4'):
            output = four
        elif(row[0] == '5'):
            output = five
        elif(row[0] == '6'):
            output = six
        elif(row[0] == '7'):
            output = seven
        elif(row[0] == '8'):
            output = eight
        elif(row[0] == '9'):
            output = nine
        del row[0]
        row = list(map(int, row))
        if(neural_network == None):
            neural_network = ConvolutionalNeuralNetwork(row, output, [30, 20, 16, 20])
        neural_network.addEntry(row, output)
csvFile.close()
with open('MNIST/mnist_test.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    print "inputting test data"
    for row in reader:
        output = None
        if(row[0] == '0'):
            output = zero
        elif(row[0] == '1'):
            output = one
        elif(row[0] == '2'):
            output = two
        elif(row[0] == '3'):
            output = three
        elif(row[0] == '4'):
            output = four
        elif(row[0] == '5'):
            output = five
        elif(row[0] == '6'):
            output = six
        elif(row[0] == '7'):
            output = seven
        elif(row[0] == '8'):
            output = eight
        elif(row[0] == '9'):
            output = nine
        del row[0]
        row = list(map(int, row))
        neural_network.addTestEntry(row, output)
csvFile.close()

print neural_network.test()
for x in range(100):
    neural_network.train(10000)
    print neural_network.test()