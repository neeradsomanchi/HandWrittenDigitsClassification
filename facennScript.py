'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
import pickle
from math import sqrt
from scipy.optimize import minimize

# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W



# Replace this with your sigmoid implementation
def sigmoid(z):
    return  1.0 / (1.0 + np.exp(-z))
# Replace this with your nnObjFunction implementation
def nnObjFunction(params, *args):
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here

    trainingData_size = training_data.shape[0]

    training_output = np.zeros((trainingData_size,n_class))

    for i in range(len(training_label)):
        training_output[i][int(training_label[i])] = 1

    nnOutputValues, hidden_output_withBias, hidden_output, input_matrix_withBias = nnOutput(w1,w2,training_data)

    t1 = np.multiply(training_output, np.log(nnOutputValues))
    t2 = np.multiply(np.subtract(1,training_output),np.log(np.subtract(1,nnOutputValues)))

    t3 = np.add(t1,t2)

    obj_val = np.divide(np.sum(t3),-1*trainingData_size)

    reg = np.sum(np.power(w1,2)) + np.sum(np.power(w2,2))

    # Value of error function
    obj_val = obj_val + np.divide(np.multiply(reg,lambdaval),2*trainingData_size)


    # Gradient matrix calculation
    delta_l = np.subtract(nnOutputValues,training_output)

    Z_j = hidden_output_withBias

    # delta_Jmatrix is a (output layer neurons * hidden layer neurons) size matrix
    delta_Jmatrix = np.dot(delta_l.T,Z_j)

    temp = np.add(delta_Jmatrix,np.multiply(lambdaval,w2))

    grad_w2 = np.divide(temp,trainingData_size)

    # We remove the last column of the hidden layer since we are moving the opposite direction, i.e, back propagation
    w2_mod = np.delete(w2,n_hidden,1)

    # Same reason as above
    Z_j = hidden_output

    t4 = np.subtract(1,Z_j)

    t5 = np.multiply(t4,Z_j)

    t6 = np.dot(delta_l,w2_mod)

    t7 = np.multiply(t5,t6)

    delta_Jmatrix = np.dot(t7.T,input_matrix_withBias)

    grad_w1 = np.divide(np.add(delta_Jmatrix,np.multiply(lambdaval,w1)),trainingData_size)

    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)

    print(obj_val)
    return (obj_val, obj_grad)

# Replace this with your nnPredict implementation
def nnPredict(w1,w2,data):
    ones_matrix = np.ones((data.shape[0],1))

    input_matrix_withBias = np.concatenate((data,ones_matrix),1)

    hidden_output = np.dot(input_matrix_withBias,w1.T)

    hidden_output = sigmoid(hidden_output)

    hidden_output_withBias = np.concatenate((hidden_output,ones_matrix),1)

    final_output = np.dot(hidden_output_withBias,w2.T)

    final_output = sigmoid(final_output)

    labels = np.argmax(final_output,axis=1)

    # labels = np.zeros((data.shape[0],1))
    # for i in range(data.shape[0]):
    #     labels[i,0] = np.argmax(final_output[i])

    return labels

def nnOutput(w1,w2,data):

    ones_matrix = np.ones((data.shape[0],1))

    input_matrix_withBias = np.concatenate((data,ones_matrix),1)

    hidden_output = np.dot(input_matrix_withBias,w1.T)

    hidden_output = sigmoid(hidden_output)

    hidden_output_withBias = np.concatenate((hidden_output,ones_matrix),1)

    final_output = np.dot(hidden_output_withBias,w2.T)

    final_output = sigmoid(final_output)

    return final_output, hidden_output_withBias, hidden_output, input_matrix_withBias 

# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 10;
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
