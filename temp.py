# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import csv as csv
import numpy as np
from mtool import nnCostFunction
from matplotlib.pyplot import plot
from mtool import predict

train_file = open('./train.csv', 'rb')
train_file_object = csv.reader(train_file)
header = train_file_object.next()

data = []
for row in train_file_object:
    data.append(row)
data = np.array(data).astype(float)

m = data.shape[0]

X = np.mat(np.hstack((np.ones((m,1)),data[::,1::])))
Y = data[::,0]

print data
print "X=",X,"Y=",Y

input_layer_size = X.shape[1] - 1
hidden_layer_size = 25
num_labels = 10
########################################################
#randinitializeWeight
########################################################

def randInitializeWeights(L_in,L_out):
    epsilon_init = 0.12
    return np.random.random(size = (L_in,L_out)) * 2 * epsilon_init - epsilon_init
    
initial_Theta1 = randInitializeWeights(input_layer_size+1, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size+1, num_labels)

lamb = 0
num_iter = 10
alpha = 0.5
J_history = []

Theta1 = initial_Theta1
Theta2 = initial_Theta2
J = float('inf')

for t in xrange(num_iter):
    print "iterator times: ", t
    
    (J,Theta1_grad,Theta2_grad) = nnCostFunction(initial_Theta1, initial_Theta2 \
                                                ,input_layer_size \
                                                ,hidden_layer_size, \
                                                num_labels,X, Y, lamb)
    print "cost :", J                                
    initial_Theta1 = initial_Theta1 - alpha * Theta1_grad
    initial_Theta2 = initial_Theta2 - alpha * Theta2_grad
    if(len(J_history) == 0 or J_history[-1] > J):
        J_history.append(float(J))
        Theta1 = initial_Theta1
        Theta2 = initial_Theta2
    else:
        break

x_axis = range(len(J_history))
plot(x_axis,J_history)

correct_in_train = 0

for row in data:
    ans = predict(Theta1,Theta2,row[1::]) + 1
    if ans == row[0]:
        correct_in_train += 1
#    print "ans is:",ans,"true val is:",row[0]

print "Train accuracy:", correct_in_train/float(m)

#test_file = open('./test.csv', 'rb')
#test_file_object = csv.reader(test_file)
#header = test_file_object.next()

#predictions_file = open('./genderclassmodel.csv', 'wb')
#predictions_file_object = csv.writer(predictions_file)
#predictions_file_object.writerow(['PassengerId','Survived'])

                           
#predictions_file.close()
#test_file.close()