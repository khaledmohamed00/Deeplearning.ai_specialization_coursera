#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 10:19:15 2019

@author: khaled
"""

import numpy as np
import matplotlib.pyplot as plt
from gradient_checking import gradient_checking



def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
    
def relu(x):
    
    s = np.maximum(0,x)
    
    return s
    
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x) *(1-sigmoid (x))

def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)


def init_weights(network):
    weights={}
    biases={}
    for l in range(len(network)-1):
        weights['w'+str(l+1)]=np.random.randn(network[l],network[l+1])*np.sqrt(2 / network[l])
        biases['b'+str(l+1)]=np.random.randn(1,network[l+1])*0.01
    return weights,biases

def forward_prop(X,weights,biases,activation_function='sigmoid'):
    L=len(weights)
    a=X
    activations=[]
    Zs=[]
    activations.append(a)

    for l in range(L-1):

        z=np.dot(a,weights['w'+str(l+1)])+biases['b'+str(l+1)]
        if activation_function=='relu':
           a=relu(z)
        elif activation_function=='sigmoid':   
           a=sigmoid(z)
        activations.append(a)
        Zs.append(z)
    
    z=np.dot(a,weights['w'+str(L)])+biases['b'+str(L)]

    a=softmax(z)
    activations.append(a)
    Zs.append(z)    
    return activations,Zs

def cost_function(a,y):
    m=a.shape[0]
    return (1.0*np.sum(-y * np.log(a)))/m

def back_prop(weights,activations,Zs,y,m,activation_function='sigmoid'):
    grad={}
    L=len(weights)
    dz = activations[-1] - y
    dw= (1./m)*np.dot(activations[-2].T, dz)
    db= (1./m)*np.sum(dz,axis=0,keepdims = True)
    grad['w'+str(L)]=dw
    grad['b'+str(L)]=db
    for i in reversed(range(1,L)):
        da= np.dot(weights['w'+str(i+1)], dz.T)
        dz = np.multiply(da.T, np.int64(activations[i] > 0))
        
        if activation_function =='relu':
           dz = np.multiply(da.T, np.int64(activations[i] > 0))
        elif activation_function =='sigmoid':
           dz =np.multiply(da.T,sigmoid_prime(Zs[i-1]))
        dw = 1./m *np.dot(activations[i-1].T,dz)
        db = 1./m *np.sum(dz, axis=0, keepdims = True)
      
        grad['w'+str(i)]=dw
        grad['b'+str(i)]=db
      
    return   grad
    
    
    
def update_params(weights,biases,grad,learning_rate):
    L=len(weights)
    for l in range(L):
      weights['w'+str(l+1)] -= learning_rate * grad['w'+str(l+1)]
      biases['b'+str(l+1)] -= learning_rate * grad['b'+str(l+1)]#.sum(axis=0)







N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j

y=y.reshape(X.shape[0],1)

m = X.shape[0]
input_dim = X.shape[1]
hidden_nodes = 4
output_labels = 3
network=[input_dim,50,20,3]
lr = 0.01
weights,biases=init_weights(network)

np.random.seed(0)
m=X.shape[0]
shuffled_indx=[np.random.permutation(m)]
X=X[shuffled_indx,:].reshape(m,-1)
y=y[shuffled_indx,:].reshape(m,-1)
y_hot=np.zeros([m,K])
for i in range(m):
    y_hot[i][y[i]]=1

#feature_set=X
one_hot_labels=y_hot

'''
error_cost = [] 

for epoch in range(5000):

    activations,Zs=forward_prop(X,weights,biases)

    grad=back_prop(weights,activations,Zs,y_hot,m)
    update_params(weights,biases,grad,lr)
        
    if epoch % 200 == 0:
        loss = cost_function(activations[-1],one_hot_labels)#np.sum(-one_hot_labels * np.log(activations[-1]))
        print('Loss function value: ', loss)
        error_cost.append(loss)


plt.plot(error_cost)
#Show the plot
plt.show()        

scores,zs = forward_prop(X,weights,biases)
predicted_class = np.argmax(scores[-1], axis=1)
print ('training accuracy: %.2f' % (np.mean(predicted_class.reshape(y.shape) == y)))
'''



activations,Zs=forward_prop(X,weights,biases)
        
grad=back_prop(weights,activations,Zs,y_hot,m)
gradient_checking(weights,biases,network, grad, X, y_hot,forward_prop,cost_function,epsilon=1e-7)
