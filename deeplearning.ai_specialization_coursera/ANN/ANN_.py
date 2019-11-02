#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 14:44:37 2019

@author: khaled
"""

import numpy as np
import matplotlib.pyplot as plt
from gradient_checking import gradient_checking


class ANN:
    def __init__(self,network,iteration=100,optimizer='SGD',regularization='none',activation_function='sigmoid',learning_rate=0.01,lambd=0.2,keep_prop=0.9,beta=0.9,batch_size=64):
        self.network=network
        self.weights,self.biases=self.init_weights(self.network)
        self.activation_function=activation_function
        self.learning_rate=learning_rate
        self.lambd=lambd
        self.keep_prop=keep_prop
        self.beta=beta
        self.batch_size=batch_size
        self.iteration=iteration
        self.optimizer=optimizer
        self.regularization=regularization
        if(self.optimizer=='adam' and self.regularization=='L2'):
           self.Vdw,self.Vdb,self.Sdw,self.Sdb=self.initialize_adam(self.weights,self.biases)
       
        elif(self.optimizer=='adam' and self.regularization=='dropout'):
           self.Vdw,self.Vdb,self.Sdw,self.Sdb=self.initialize_adam(self.weights,self.biases)
    
        elif(self.optimizer=='adam' and self.regularization=='none'):
           self.Vdw,self.Vdb,self.Sdw,self.Sdb=self.initialize_adam(self.weights,self.biases)    
    
        elif(self.optimizer=='momentum' and self.regularization=='L2'):
           self.Vdw,self.Vdb=self.initialize_velocity(self.weights,self.biases)
        elif(self.optimizer=='momentum' and self.regularization=='dropout'):
           self.Vdw,self.Vdb=self.initialize_velocity(self.weights,self.biases)
        elif(self.optimizer=='momentum' and self.regularization=='none'):
           self.Vdw,self.Vdb=self.initialize_velocity(self.weights,self.biases)
        
    def init_weights(self,network):
        weights={}
        biases={}
        for l in range(len(network)-1):
            weights['w'+str(l+1)]=np.random.randn(network[l],network[l+1])*np.sqrt(2 / network[l])
            biases['b'+str(l+1)]=np.random.randn(1,network[l+1])*0.01
        return weights,biases
    def initialize_velocity(self,weights,biases):
        L=len(weights)
        Vdw={}
        Vdb={}
        for l in range(L):
           Vdw['dw'+str(l+1)]=np.zeros_like(weights['w'+str(l+1)])
           Vdb['db'+str(l+1)]=np.zeros_like(biases['b'+str(l+1)])
        return Vdw,Vdb            


    def initialize_adam(self,weights,biases) :
        
        Vdw,Vdb =self.initialize_velocity(weights,biases)
        Sdw,Sdb =self.initialize_velocity(weights,biases)
    
        return  Vdw,Vdb,Sdw,Sdb
    
    
    def forward_prop(self,X):
        L=len(self.weights)
        a=X
        activations=[]
        Zs=[]
        activations.append(a)

        for l in range(L-1):

            z=np.dot(a,self.weights['w'+str(l+1)])+self.biases['b'+str(l+1)]
            if self.activation_function=='relu':
               a=self.relu(z)
            elif self.activation_function=='sigmoid':   
               a=self.sigmoid(z)
            activations.append(a)
            Zs.append(z)
    
        z=np.dot(a,self.weights['w'+str(L)])+self.biases['b'+str(L)]

        a=self.softmax(z)
        activations.append(a)
        Zs.append(z)    
        return activations,Zs



    def forward_prop_with_dropout(self,X):
        L=len(self.weights)
        a=X
        activations=[]
        Zs=[]
        Ds=[]
        activations.append(a)

        for l in range(L-1):

            z=np.dot(a,self.weights['w'+str(l+1)])+self.biases['b'+str(l+1)]
            if self.activation_function=='relu':
               a=self.relu(z)
            elif self.activation_function=='sigmoid':   
               a=self.sigmoid(z)
            D = np.random.rand(a.shape[0], a.shape[1])
            D=D<self.keep_prop
            Ds.append(D)
            a=a*D/self.keep_prop
            #a=sigmoid(z)
            activations.append(a)
            Zs.append(z)
    
        z=np.dot(a,self.weights['w'+str(L)])+self.biases['b'+str(L)]
        
        a=self.sigmoid(z)
        activations.append(a)
        Zs.append(z)    
        return activations,Zs,Ds
    
    def compute_loss(self,a,y):
        m=a.shape[0]
        return (1.0*np.sum(-y * np.log(a)))/m  
    
    
    def compute_cost_with_regularization(self,a, Y, lambd):
        
        m = Y.shape[0]
        L=len(self.weights)
        cross_entropy_cost=self.compute_loss(a, Y)
   
        sum_weights=0 
        for i in range(L):
           sum_weights =sum_weights+np.sum(np.square(self.weights['w'+str(i+1)]))
      
        l2_regularization_cost=lambd * sum_weights/ (2 * m)
   
        cost=cross_entropy_cost+l2_regularization_cost
   
        return cost
    

    
    
    
    def back_prop(self,activations,Zs,y,m):
        grad={}
        L=len(self.weights)
        dz = activations[-1] - y
        dw= (1./m)*np.dot(activations[-2].T, dz)
        db= (1./m)*np.sum(dz,axis=0,keepdims = True)
        grad['w'+str(L)]=dw
        grad['b'+str(L)]=db
        for i in reversed(range(1,L)):
            da= np.dot(self.weights['w'+str(i+1)], dz.T)
        
            if self.activation_function =='relu':
               dz = np.multiply(da.T, np.int64(activations[i] > 0))
            elif self.activation_function =='sigmoid':
               dz =np.multiply(da.T,self.sigmoid_prime(Zs[i-1]))
            dw = 1./m *np.dot(activations[i-1].T,dz)
            db = 1./m *np.sum(dz, axis=0, keepdims = True)
      
            grad['w'+str(i)]=dw
            grad['b'+str(i)]=db
      
        return   grad
    
    
    
    
    
    
    def back_prop_with_regularization(self,activations,Zs,y,m):
        L=len(self.weights)

        grad={}
        dz = activations[-1] - y
        dw = 1./m *np.dot(activations[-2].T,dz)+(self.lambd * self.weights['w'+str(L)]) / m
        db = 1./m *np.sum(dz, axis=0, keepdims = True)
        grad['w'+str(L)]=dw
        grad['b'+str(L)]=db
    

        for i in reversed(range(1,L)):
            da= np.dot(self.weights['w'+str(i+1)], dz.T)
        
            if self.activation_function =='relu':
               dz = np.multiply(da.T, np.int64(activations[i] > 0))
            elif self.activation_function =='sigmoid':
               dz =np.multiply(da.T,self.sigmoid_prime(Zs[i-1]))
            dw = 1./m *np.dot(activations[i-1].T,dz)+(self.lambd * self.weights['w'+str(i)]) / m
            db = 1./m *np.sum(dz, axis=0, keepdims = True)
      
            grad['w'+str(i)]=dw
            grad['b'+str(i)]=db
    
        return   grad



    def back_prop_with_dropout(self,activations,Zs,Ds,y,m):
        grad={}
        dz =  activations[-1] - y
        dw = 1./m *np.dot(activations[-2].T,dz)
        db = 1./m *np.sum(dz, axis=0, keepdims = True)
        L=len(self.weights)
        grad['w'+str(L)]=dw
        grad['b'+str(L)]=db
    
        for i in reversed(range(1,L)):
            da= np.dot(self.weights['w'+str(i+1)], dz.T)
            da=da*Ds[i-1].T
            if self.activation_function =='relu':
               dz = np.multiply(da.T, np.int64(activations[i] > 0))
            elif self.activation_function =='sigmoid':
               dz =np.multiply(da.T,self.sigmoid_prime(Zs[i-1]))
            dw = 1./m *np.dot(activations[i-1].T,dz)
            db = 1./m *np.sum(dz, axis=0, keepdims = True)
      
            grad['w'+str(i)]=dw
            grad['b'+str(i)]=db
      
        return   grad
    
    
    
        
    
    
    
    def update_params(self,grad):
        L=len(self.weights)
        for l in range(1,L):
            self.weights['w'+str(l+1)]=self.weights['w'+str(l+1)]-self.learning_rate*grad['w'+str(l+1)]
            self.biases['b'+str(l+1)]=self.biases['b'+str(l+1)]-self.learning_rate*grad['b'+str(l+1)]
    


    def update_params_with_momentum(self,grad):
        L=len(self.weights)
        for l in range(L):
           self.Vdw['dw'+str(l+1)]=self.beta*self.Vdw['dw'+str(l+1)]+(1-self.beta)*grad['w'+str(l+1)]
           self.Vdb['db'+str(l+1)]=self.beta*self.Vdb['db'+str(l+1)]+(1-self.beta)*grad['b'+str(l+1)]
    
           self.weights['w'+str(l+1)]=self.weights['w'+str(l+1)]-self.learning_rate*self.Vdw['dw'+str(l+1)]
           self.biases['b'+str(l+1)]=self.biases['b'+str(l+1)]-self.learning_rate*self.Vdb['db'+str(l+1)]


    def update_parameters_with_adam(self,grads,t,
                                beta1=0.9, beta2=0.999, epsilon=1e-8):
         Vdw_corrected = {}                         # Initializing first moment estimate, python dictionary
         Vdb_corrected = {}   
         Sdw_corrected = {}
         Sdb_corrected = {}
    
         L=len(self.weights)
         for l in range(L):
        
             self.Vdw['dw'+str(l+1)] = beta1 * self.Vdw['dw'+str(l+1)] + (1 - beta1) * grads['w'+ str(l + 1)]
             self.Vdb['db' + str(l + 1)] = beta1 * self.Vdb['db' + str(l + 1)] + (1 - beta1) * grads['b' + str(l + 1)]
             Vdw_corrected['dw'+str(l+1)] = self.Vdw['dw'+str(l+1)] / (1 - np.power(beta1, t))
             Vdb_corrected['db' + str(l + 1)] = self.Vdb['db' + str(l + 1)] / (1 - np.power(beta1, t))
             self.Sdw['dw'+str(l+1)] = beta2 * self.Sdw['dw'+str(l+1)] + (1 - beta2) * np.power(grads['w' + str(l + 1)], 2)
             self.Sdb['db' + str(l + 1)] = beta2 * self.Sdb['db' + str(l + 1)] + (1 - beta2) * np.power(grads['b' + str(l + 1)], 2)
             Sdw_corrected['dw'+str(l+1)] = self.Sdw['dw'+str(l+1)] / (1 - np.power(beta2, t))
             Sdb_corrected['db' + str(l + 1)] = self.Sdb['db' + str(l + 1)] / (1 - np.power(beta2, t))
             self.weights['w'+ str(l + 1)] = self.weights['w' + str(l + 1)] - self.learning_rate * Vdw_corrected['dw' + str(l + 1)] / np.sqrt(Sdw_corrected['dw' + str(l + 1)] + epsilon)
             self.biases['b' + str(l + 1)] = self.biases['b' + str(l + 1)] - self.learning_rate * Vdb_corrected['db' + str(l + 1)] / np.sqrt(Sdb_corrected['db' + str(l + 1)] + epsilon)

    
    
 
    
    def gradient_descenet(self,X,y,m):
        activations,Zs  =self.forward_prop(X)
        
        grad=self.back_prop(activations,Zs,y,m)
        
        self.update_params(grad)

        loss=self.compute_loss(activations[-1], y)
    
        return loss

    def gradient_descenet_with_momentum(self,X,y,m):
        activations,Zs  =self.forward_prop(X)
         
        grad=self.back_prop(activations,Zs,y,m)
        
        self.update_params_with_momentum(grad)    

        loss=self.compute_loss(activations[-1], y)
    
        return loss

    def gradient_descenet_with_adam(self,X,y,t,m):
         activations,Zs  =self.forward_prop(X)
        
         grad=self.back_prop(activations,Zs,y,m)
    
         self.update_parameters_with_adam(grad,2,beta1=0.9, beta2=0.99, epsilon=1e-8)    

         loss=self.compute_loss(activations[-1], y)
    
         return loss

    def gradient_descenet_with_regularization(self,X,y,m):
        activations,Zs  =self.forward_prop(X)
        grad=self.back_prop_with_regularization(activations,Zs,y,m)
        
        self.update_params(grad)
        loss=self.compute_cost_with_regularization(activations[-1], y)
       
        return loss

    def gradient_descenet_with_momentum_with_regularization(self,X,y,m):
        activations,Zs  =self.forward_prop(X)
        grad=self.back_prop_with_regularization(activations,Zs,y,m)
        
        self.update_params_with_momentum(grad)
        loss=self.compute_cost_with_regularization(activations[-1], y)
       
        return loss

    def gradient_descenet_with_adam_with_regularization(self,X,y,t,m):
        activations,Zs  =self.forward_prop(X)
        grad=self.back_prop_with_regularization(activations,Zs,y,m)
    
    #grad=back_prop(weights,activations,Zs,y,m)
    
        self.update_parameters_with_adam(grad,2,beta1=0.9, beta2=0.99, epsilon=1e-8)    
    #update_params(weights,biases,grad,learning_rate)

        loss=self.compute_loss(activations[-1], y)
    
        return loss
    
    def gradient_descenet_with_Dropout(self,X,y,m):
       activations,Zs,Ds=self.forward_prop_with_dropout(X)
       grad=self.back_prop_with_dropout(activations,Zs,Ds,y,m)
        
       self.update_params(grad)
       loss=self.compute_loss(activations[-1], y)
    
       return loss

    def gradient_descenet_with_momentum_with_Dropout(self,X,y,m):
        activations,Zs,Ds=self.forward_prop_with_dropout(X)
        grad=self.back_prop_with_dropout(activations,Zs,Ds,y,m)
        
        self.update_params_with_momentum(grad)    
        loss=self.compute_loss(activations[-1], y)
    
        return loss

    def gradient_descenet_with_adam_with_Dropout(self,X,y,t,m):
        activations,Zs,Ds=self.forward_prop_with_dropout(X)
        grad=self.back_prop_with_dropout(activations,Zs,Ds,y,m)
    
        self.update_parameters_with_adam(grad,2,beta1=0.9, beta2=0.99, epsilon=1e-8)    
    #update_params(weights,biases,grad,learning_rate)

        loss=self.compute_loss(activations[-1], y)
    
        return loss

    
    def random_mini_batch(self,X,y,seed=0):
        mini_batches=[]
        np.random.seed(0)
        m=X.shape[0]
        shuffled_indx=[np.random.permutation(m)]
        X_shuffled=X[shuffled_indx,:].reshape(m,-1)
        y_shuffled=y[shuffled_indx,:].reshape(m,-1)
        no_complete_batches=(int)(m/self.batch_size)
        for i in range(no_complete_batches):
            X_batch=X_shuffled[i*self.batch_size:(i+1)*self.batch_size,:]
            y_batch=y_shuffled[i*self.batch_size:(i+1)*self.batch_size,:]
            mini_batches.append((X_batch,y_batch))
     
        if(m%self.batch_size!=0):
           X_batch=X_shuffled[no_complete_batches*self.batch_size:,:]
           y_batch=y_shuffled[no_complete_batches*self.batch_size:,:]
           mini_batches.append((X_batch,y_batch))
    
        return mini_batches
    
    
    
    def mini_batch_SGD(self,mini_batches):
        losses=[]
        
        for i in range(self.iteration):
            c=0
            for batch in mini_batches:
                loss=self.gradient_descenet(batch[0],batch[1],batch[0].shape[0])
                losses.append(loss)
                print("epoch : %5d, batch : %5d loss : %5.5f"%(i+1,c+1,loss))
                c+=1

        return losses        


    def mini_batch_SGD_with_regularization(self,mini_batches):
        losses=[]
        for i in range(self.iteration):
           c=0
           for batch in mini_batches:
               loss=self.gradient_descenet_with_regularization(batch[0],batch[1],batch[0].shape[0])
               losses.append(loss)
               print("epoch : %5d, batch : %5d loss : %5.5f"%(i+1,c+1,loss))
               c+=1

        return losses    

    def mini_batch_SGD_with_dropout(self,mini_batches):
        losses=[]
        for i in range(self.iteration):
           c=0

           for batch in mini_batches:
               loss=self.gradient_descenet_with_Dropout(batch[0],batch[1],batch[0].shape[0])
               losses.append(loss)
               print("epoch : %5d, batch : %5d loss : %5.5f"%(i+1,c+1,loss))
               c+=1

        return losses    




    def mini_batch_SGD_with_adam(self,mini_batches):
        losses=[]
        for i in range(self.iteration):
           c=0
           for batch in mini_batches:
               loss=self.gradient_descenet_with_adam(batch[0],batch[1],batch[0].shape[0])
               losses.append(loss)
               print("epoch : %5d, batch : %5d loss : %5.5f"%(i+1,c+1,loss))
               c+=1

        return losses 


    def mini_batch_SGD_with_adam_with_regularization(self,mini_batches,t):
        losses=[]
        
        for i in range(self.iteration):
            c=0
            for batch in mini_batches:
                loss=self.gradient_descenet_with_adam_with_regularization(batch[0],batch[1],t,batch[0].shape[0])
                losses.append(loss)
                print("epoch : %5d, batch : %5d loss : %5.5f"%(i+1,c+1,loss))
                c+=1

        return losses 



    def mini_batch_SGD_with_adam_with_dropout(self,mini_batches,t):
        losses=[]
        for i in range(self.iteration):
            c=0
            for batch in mini_batches:
                loss=self.gradient_descenet_with_adam_with_Dropout(batch[0],batch[1],t,batch[0].shape[0])
                losses.append(loss)
                print("epoch : %5d, batch : %5d loss : %5.5f"%(i+1,c+1,loss))
                c+=1

        return losses 




    def mini_batch_SGD_with_momemtum(self,mini_batches):
        losses=[]
        for i in range(self.iteration):
            c=0
            for batch in mini_batches:
                loss=self.gradient_descenet_with_momentum(batch[0],batch[1],batch[0].shape[0])
                losses.append(loss)
                print("epoch : %5d, batch : %5d loss : %5.5f"%(i+1,c+1,loss))
                c+=1

        return losses 


    def mini_batch_SGD_with_momemtum_with_regularization(self,mini_batches):
        losses=[]
        for i in range(self.iteration):
            c=0
            for batch in mini_batches:
                loss=self.gradient_descenet_with_momentum_with_regularization(batch[0],batch[1],batch[0].shape[0])
                losses.append(loss)
                print("epoch : %5d, batch : %5d loss : %5.5f"%(i+1,c+1,loss))
                c+=1

        return losses 



    def mini_batch_SGD_with_momentum_with_dropout(self,mini_batches):
        losses=[]
        for i in range(self.iteration):
            c=0
            for batch in mini_batches:
                loss=self.gradient_descenet_with_momentum_with_Dropout(batch[0],batch[1],batch[0].shape[0])
                losses.append(loss)
                print("epoch : %5d, batch : %5d loss : %5.5f"%(i+1,c+1,loss))
                c+=1

        return losses 
    
    
    
    def fit(self,X,y):  
        mini_batches=self.random_mini_batch(X,y,seed=0)
    
        if(self.optimizer=='adam' and self.regularization=='L2'):
           loss=self.mini_batch_SGD_with_adam_with_regularization(mini_batches,2)
       
        elif(self.optimizer=='adam' and self.regularization=='dropout'):
           loss=self.mini_batch_SGD_with_adam_with_dropout(mini_batches,2)
    
        elif(self.optimizer=='adam' and self.regularization=='none'):
           loss=self.mini_batch_SGD_with_adam(mini_batches,2)
    
        elif(self.optimizer=='momentum' and self.regularization=='L2'):
           loss=self.mini_batch_SGD_with_momemtum_with_regularization(mini_batches)
        elif(self.optimizer=='momentum' and self.regularization=='dropout'):
           loss=self.mini_batch_SGD_with_momentum_with_dropout(mini_batches)
        elif(self.optimizer=='momentum' and self.regularization=='none'):
            loss=self.mini_batch_SGD_with_momemtum(mini_batches)
       
    
        elif(self.optimizer=='SGD' and self.regularization=='L2'):
           loss=self.mini_batch_SGD_with_regularization(mini_batches)
        elif(self.optimizer=='SGD' and self.regularization=='dropout'):
           loss=self.mini_batch_SGD_with_dropout(mini_batches)
        elif(self.optimizer=='SGD' and self.regularization=='none'):   
           loss=self.mini_batch_SGD(mini_batches)
        return loss
          
    def evaluate(self,X,y):
        scores,zs = self.forward_prop(X)
        predicted_class = np.argmax(scores[-1], axis=1)
        
        print ('training accuracy: %.2f' % (np.mean(predicted_class.reshape(y.shape) == y)))
        return scores[-1]
        
    def predict(self,X):
        scores,zs = self.forward_prop(X)
        predicted_class = np.argmax(scores[-1], axis=1)
        
       
        return predicted_class

       
    def sigmoid_prime(self,z):

        return self.sigmoid(z)*(1-self.sigmoid(z))
    
    def relu(self,x):
    
        s = np.maximum(0,x)
    
        return s
    
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))



    def softmax(self,z):
        expz = np.exp(z)
        return expz / expz.sum(axis=1, keepdims=True)
    
    


def load_data():    
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
    m = X.shape[0]
    y=y.reshape([m,1])
    np.random.seed(0)
    shuffled_indx=[np.random.permutation(m)]
    X=X[shuffled_indx,:].reshape(m,-1)
    y=y[shuffled_indx,:].reshape(m,-1)
    y_hot=np.zeros([m,K])
    for i in range(m):
        y_hot[i][y[i]]=1

    return X,y_hot,y
    


X,y_hot,y=load_data()
network=[2,50,20,3]
m=300 
learning_rate=0.01
lambd=0.2
keep_prop=0.9
beta=0.9
iteration=100
batch_size=64 

net=ANN(network,iteration=300,optimizer='adam',regularization='L2',activation_function='relu',learning_rate=0.1,lambd=0.2,keep_prop=0.9,beta=0.9,batch_size=64)    
loss=net.fit(X,y_hot)
plt.plot(loss)
#Show the plot
plt.show() 
net.evaluate(X,y)
#scores,zs = net.forward_prop(X)
#predicted_class = np.argmax(scores[-1], axis=1)
#print ('training accuracy: %.2f' % (np.mean(predicted_class.reshape(y.shape) == y)))