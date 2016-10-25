'''
Main Executable program for BackPropNetwork.
Takes care of data preparation, and initlization of the BackPropNetwork class
'''
'''
12jjm4@queensu.ca
John Martin
10084202
'''

import numpy as np

from BackPropNetwork import BackPropNetwork

#Data preparation class
def get_data(f):
    data = np.loadtxt(f, delimiter = ',')
    #Empty array for predicted (whole integer) values
    actualNum = []
    for i in range(len(data)):
        tmp = np.zeros(10)
        tmp[int(data[i][64])] = 1
        actualNum.append(tmp)
    #delete predicted values from matrix (last column)
    data = np.delete(data, (64), axis=1)
    data -= data.min()
    data /= data.max()
    return (data, actualNum)

data = get_data('Data-Set/training.txt')
inputData = data[0]
actualNums = data[1]

#See README for explaination of value choices
bp = BackPropNetwork(input_sz = 64, hidden_sz = 30, output_sz = 10, learning_rate = 0.7, momentum = 0.5)


#Feel free to experiment with number of iterations
bp.train(inputData, actualNums, iterations = 2)

#Change dataset to the testing data
data = get_data('Data-Set/testing.txt')
inputData = data[0]
actualNum = data[1]
#test input data
bp.test(inputData, actualNum)
