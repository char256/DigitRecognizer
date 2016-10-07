# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 20:11:55 2016

@author: xzx
"""
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
    
    
def nnCostFunction(Theta1,Theta2,\
                    input_layer_size,hidden_layer_size, \
                    num_labels, X, y, lamb):

    m = float(X.shape[0])
    a2 = np.mat(np.hstack((np.ones((m,1)),sigmoid(X * Theta1))))
    a3 = sigmoid(a2 * Theta2)
    Y = np.zeros((int(m),num_labels))
    
    for i in xrange(0,int(m)):
        Y[i][int(y[i])] = 1.0
        
    J = -1/m * sum(sum(np.multiply(Y , np.log(a3)) + np.multiply((1 - Y) , np.log(1 - a3)))[:,0]) \
        + lamb/(2*m) * ( np.multiply(Theta1[::,1::],Theta1[::,1::]).sum() + np.multiply(Theta2[::,1::], Theta2[::,1::]).sum() )
        
    Delta1 = np.zeros(Theta1.shape)
    Delta2 = np.zeros(Theta2.shape)
    
    for t in xrange(0,int(m)):
#        print "couting:", t
        a1 = X[t,::]
        a2 = np.mat(np.hstack((np.ones((1,1)),sigmoid(a1 * Theta1))))
        a3 = sigmoid(a2 * Theta2)
        delta3 = a3 - Y[t,::];
        Delta2 = Delta2 + (delta3.T * a2).T
        delta2 = np.multiply(np.multiply((delta3 * Theta2.T) , a2)\
                                        , (1 - a2))
        Delta1 = Delta1 + (delta2[0, 1::].T * a1).T
        
    Theta1_grad = (Delta1 + lamb * Theta1)/m
    Theta2_grad = (Delta2 + lamb * Theta2)/m
    return J,Theta1_grad,Theta2_grad
    
    
def predict(Theta1, Theta2, X):
    """
    X should be original data(no bias unit)    
    """
    X = np.mat(np.hstack((np.mat(1),np.mat(X))))
    h1 = np.mat(sigmoid(X * Theta1))
    h2 = sigmoid( np.mat(np.hstack((np.mat(1),np.mat(h1)))) * Theta2)
    ########
    #get the max in h2
    return np.argmax(h2)