#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 22:29:41 2019

@author: khaled
"""

import numpy as np
import matplotlib.pyplot as plt
from gradient_checking import gradient_checking
def relu(x):
    
    s = np.maximum(0,x)
    
    return s

def softmax(z):
    exp_z = np.exp(z)
    probs = exp_z / (np.sum(exp_z, axis=1, keepdims=True))
    return np.array(probs)

def stable_softmax(x):
    z = x - np.max(x,axis=0)
    numerator = np.exp(z)
    denominator = np.sum(numerator, axis=1, keepdims=True)
    softmax = numerator/denominator
    return softmax

def sigmoid(x):
    
    s = 1/(1+np.exp(-x))
    return s

def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.  Note that np.nan_to_num is used to ensure numerical
        stability.  In particular, if both ``a`` and ``y`` have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0).

        """
        return np.sum(-y * np.log(a))#np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

def compute_loss(a, Y):

    m = Y.shape[0]
    logprobs = np.multiply(-np.log(a),Y) + np.multiply(-np.log(1 - a), 1 - Y)
    loss = 1./m * np.nansum(logprobs)
    
    return loss


def cross_entropy_loss(a,y):
    m=y.shape[0]
    correct_logprobs = -np.log(a[range(m),y])
    loss = np.sum(correct_logprobs)/m
    return loss

def init_weights(network):
    weights={}
    biases={}
    for l in range(len(network)-1):
        weights['w'+str(l+1)]=np.random.randn(network[l],network[l+1])*np.sqrt(2 / network[l])
        biases['b'+str(l+1)]=np.random.randn(1,network[l+1])*0.01
    return weights,biases


def forward_prop(X,weights,biases):
    L=len(weights)
    a=X
    activations=[]
    Zs=[]
    activations.append(a)
    for l in range(L-1):
        #print(weights['w'+str(l+1)].shape)
        #print(biases['b'+str(l+1)].shape)
        #print(z.shape)
        z=np.dot(a,weights['w'+str(l+1)])+biases['b'+str(l+1)]
        
        a=relu(z)
        #a=sigmoid(z)
        activations.append(a)
        Zs.append(z)
    
    z=np.dot(a,weights['w'+str(L)])+biases['b'+str(L)]
    #a=sigmoid(z)    
    #a=softmax(z)
    a=stable_softmax(z)
    activations.append(a)
    Zs.append(z)    
    return activations,Zs


def back_prop(weights,activations,Zs,y,m):
    grad={}
    #dz=activations[-1]
    #print(dz.shape)
    #print(m)
    #dz=np.copy(activations[-1])
    #dz[range(m),y] -= 1
    dz = activations[-1] - y
    dw = (1./m) *np.dot(activations[-2].T,dz)
    db = (1./m) *np.sum(dz, axis=0, keepdims = True)
    L=len(weights)
    grad['w'+str(L)]=dw
    grad['b'+str(L)]=db
    
    for i in reversed(range(1,L)):
       da= np.dot(weights['w'+str(i+1)], dz.T)
       dz = np.multiply(da.T, np.int64(activations[i] > 0))
       #dz = 1./m *np.multiply(da.T,sigmoid_prime(Zs[i-1]))
       dw = (1./m) *np.dot(activations[i-1].T,dz)
       db = (1./m) *np.sum(dz, axis=0, keepdims = True)
      
       grad['w'+str(i)]=dw
       grad['b'+str(i)]=db
      
    return   grad
'''
def forward_prop(X,weights,biases):
    L=len(weights)
    a=X
    activations=[]
    Zs=[]
    activations.append(a)
    for l in range(L-1):
        #print(weights['w'+str(l+1)].shape)
        #print(biases['b'+str(l+1)].shape)
        #print(z.shape)
        z=np.dot(a,weights['w'+str(l+1)])+biases['b'+str(l+1)]
        
        a=relu(z)
        #a=sigmoid(z)
        activations.append(a)
        Zs.append(z)
    
    z=np.dot(a,weights['w'+str(L)])+biases['b'+str(L)]
        
    a=sigmoid(z)
    activations.append(a)
    Zs.append(z)    
    return activations,Zs



def back_prop(weights,activations,Zs,y,m):
    grad={}
    dz =  (activations[-1] - y)
    dw = 1./m *np.dot(activations[-2].T,dz)
    db = 1./m *np.sum(dz, axis=0, keepdims = True)
    L=len(weights)
    grad['w'+str(L)]=dw
    grad['b'+str(L)]=db
    
    for i in reversed(range(1,L)):
       da= np.dot(weights['w'+str(i+1)], dz.T)
       dz = np.multiply(da.T, np.int64(activations[i] > 0))
       #dz = 1./m *np.multiply(da.T,sigmoid_prime(Zs[i-1]))
       dw = 1./m *np.dot(activations[i-1].T,dz)
       db = 1./m *np.sum(dz, axis=0, keepdims = True)
      
       grad['w'+str(i)]=dw
       grad['b'+str(i)]=db
      
    return   grad
'''

def update_params(weights,biases,grad,learning_rate):
    L=len(weights)
    for l in range(1,L):
        weights['w'+str(l+1)]=weights['w'+str(l+1)]-learning_rate*grad['w'+str(l+1)]
        biases['b'+str(l+1)]=biases['b'+str(l+1)]-learning_rate*grad['b'+str(l+1)]

def gradient_descenet(X,y,weights,biases,learning_rate,m):
    activations,Zs  =forward_prop(X,weights,biases)
        
    grad=back_prop(weights,activations,Zs,y,m)
        
    update_params(weights,biases,grad,learning_rate)

    loss=fn(np.array(activations[-1]), y)
    #loss=compute_loss(np.array(activations[-1]), y)

    return loss


def random_mini_batch(X,y,batch_size,seed=0):
    mini_batches=[]
    np.random.seed(0)
    m=X.shape[0]
    shuffled_indx=[np.random.permutation(m)]
    X_shuffled=X[shuffled_indx,:].reshape(m,-1)
    y_shuffled=y[shuffled_indx,:].reshape(m,-1)
    no_complete_batches=(int)(m/batch_size)
    for i in range(no_complete_batches):
         X_batch=X_shuffled[i*batch_size:(i+1)*batch_size,:]
         y_batch=y_shuffled[i*batch_size:(i+1)*batch_size,:]
         mini_batches.append((X_batch,y_batch))
     
    if(m%batch_size!=0):
         X_batch=X_shuffled[no_complete_batches*batch_size:,:]
         y_batch=y_shuffled[no_complete_batches*batch_size:,:]
         mini_batches.append((X_batch,y_batch))
    
    return mini_batches

def mini_batch_SGD(mini_batches,batch_size,weights,biases,learning_rate,iteration):
    losses=[]
    t=0
    for i in range(iteration):
        c=0
        for batch in mini_batches:
            loss=gradient_descenet(batch[0],batch[1],weights,biases,learning_rate,batch[0].shape[0])
            losses.append(loss)
            print("epoch : %5d, batch : %5d loss : %5.5f"%(i+1,c+1,loss))
            c+=1
            t+t+1

    return losses   



network=[2,50,20,3]
m=300 
learning_rate=0.1
lambd=0.2
keep_prop=0.9
beta=0.9
iteration=50
batch_size=64 



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
  

np.random.seed(0)
m=X.shape[0]
shuffled_indx=[np.random.permutation(m)]
X=X[shuffled_indx,:].reshape(m,-1)
y=y[shuffled_indx,:].reshape(m,-1)
y_hot=np.zeros([m,K])
for i in range(m):
    y_hot[i][y[i]]=1

#mini_batches=random_mini_batch(X,y_hot,batch_size,seed=0)
weights,biases=init_weights(network)
#loss=mini_batch_SGD(mini_batches,batch_size,weights,biases,learning_rate,iteration)
loss=[]
for i in range(iteration):
     lo=gradient_descenet(X,y,weights,biases,learning_rate,m)
     loss.append(lo)

    


plt.plot(loss)
# Show the plot
plt.show()


# lets visualize the data:
#plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
#plt.show()

#activations,Zs=forward_prop(X,weights,biases)
        
#grad=back_prop(weights,activations,Zs,y_hot,m)
#gradient_checking(weights,biases,network, grad, X, y_hot,forward_prop,fn,epsilon=1e-7)        
#update_params(weights,biases,grad,learning_rate)

#loss=fn(np.array(activations[-1]), y_hot)
