'''
12jjm4@queensu.ca
John Martin
10084202
'''

import math
import random
import numpy as np

#calculating sutiable range for weight values
def get_range(size):
    return (1.1 / size ** (1/2))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dv_sigmoid(y):
    return y * (1.0 - y)

class BackPropNetwork(object):
    #initializing input information for BackPropNetwork
    #size of each layer, learning rate, and momentum are all included
    def __init__(self, input_sz, hidden_sz, output_sz, momentum, learning_rate):
        self.input_sz = input_sz + 1
        self.hidden_sz = hidden_sz
        self.output_sz = output_sz
        self.learning_rate = learning_rate
        self.momentum = momentum

        #Creating vectors for neural network - activations, random weights, and tmp arrays
        self.weights_input = np.random.normal(loc = 0, scale = get_range(input_sz), size = (self.input_sz, self.hidden_sz))
        self.weights_output = np.random.normal(loc = 0, scale = get_range(hidden_sz), size = (self.hidden_sz, self.output_sz))
        self.val_input = [0] * self.input_sz
        self.val_hidden = [0] * self.hidden_sz
        self.val_output = [0] * self.output_sz
        self.change_in = np.zeros((self.input_sz, self.hidden_sz))
        self.change_out = np.zeros((self.hidden_sz, self.output_sz))

    def feedForward(self, x_vector):
        #Set input values as x_vector
        for i in range(self.input_sz -1):
            self.val_input[i] = x_vector[i]
        #Calculate sum at each input node, convert it to sigmoid function, move to hidden nodes
        #using Numpy built in for all arithmetic
        for j in range(self.hidden_sz):
            sum = 0
            for i in range(self.input_sz):
                sum += self.val_input[i] * self.weights_input[i][j]
            self.val_hidden[j] = sigmoid(sum)

        #Calculate sum at each hidden node, convert it to sigmoid function, move to outer nodes
        for l in range(self.output_sz):
            sum = 0
            for m in range(self.hidden_sz):
                sum += self.val_hidden[m] * self.weights_output[m][l]
            self.val_output[l] = sigmoid(sum)
        #return the output nodes
        return self.val_output[:]

    def backPropagate(self, actualNum):
        #Begin the process of backprogation
        op_vector = [0] * self.output_sz
        #calculate error based of the previous output vector, and the expected number
        for k in range(self.output_sz):
            e = -(actualNum[k] - self.val_output[k])
            op_vector[k] = dv_sigmoid(self.val_output[k]) * e


        hid_vector = [0] * self.hidden_sz
        for j in range(self.hidden_sz):
            e = 0
            for k in range(self.output_sz):
                e += op_vector[k] * self.weights_output[j][k]
            hid_vector[j] = dv_sigmoid(self.val_hidden[j]) * e

        #Next is changing the weight for both the hidden nodes and the input notes
        # (changing the weights_output and weights input matrix)
        for j in range(self.hidden_sz):
            for k in range(self.output_sz):
                change = op_vector[k] * self.val_hidden[j]
                self.weights_output[j][k] -= self.learning_rate * change + self.change_out[j][k] * self.momentum
                self.change_out[j][k] = change

        for i in range(self.input_sz):
            for j in range(self.hidden_sz):
                change = hid_vector[j] * self.val_input[i]
                self.weights_input[i][j] -= self.learning_rate * change + self.change_in[i][j] * self.momentum
                self.change_in[i][j] = change

        # calculate the root mean square error (to see the progress that our algorithm is making during each iteration)
        #this value is just printed after every iteration
        error = np.zeros(len(actualNum))
        for k in range(len(actualNum)):
            error[k] =  0.5 * (actualNum[k] - self.val_output[k]) ** 2
        mean = np.mean(error)
        e = math.sqrt(mean)
        return e

    #interprets final output vector
    def ff_return(self, a):
        m = max(a)
        new_a = a
        for i in range(len(a)):
            if m == a[i]:
                new_a[i]=1
            else:
                new_a[i]=0
        return new_a


    def test(self, inputData, actualNum):
        f = open('test-results.txt', 'w')
        incorrect = len(inputData)
        for i in range(len(inputData)):
            #use existing feedForward network to predict the letter
            output = self.feedForward(inputData[i])
            #keep track of the number of outputs guessed wrong
            if not np.array_equal(self.ff_return(output), actualNum[i]):
                incorrect -= 1
            #write out a subset of the data
            if i % 15 ==0:
                f.write('Predicted: {0} ---> Actual: {1}\n'.format(actualNum[i], self.ff_return(output)))
        f.write('\n\nTotal accuracy on test sample: {0}'.format(float(incorrect)/float(len(inputData))))
        f.close()
        return

#traing class, calls feedForward and back propgate for a given number of iterations
    def train(self, inputData, actualNum, iterations):
        for i in range(iterations):
            error = 0
            for j in range(len(inputData)):
                self.feedForward(inputData[j])
                error += self.backPropagate(actualNum[j])
            print "RMSE: {0}".format(error)
