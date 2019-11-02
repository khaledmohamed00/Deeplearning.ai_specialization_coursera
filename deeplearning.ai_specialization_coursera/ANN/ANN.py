import numpy as np
from init_utils import load_dataset,plot_decision_boundary
from reg_utils import  load_2D_dataset
import matplotlib.pyplot as plt
from gradient_checking import gradients_to_vector,dirctory_to_vector,vector_to_directory,gradient_checking



def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    s = 1/(1+np.exp(-x))
    return s

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def relu(x):
    """
    Compute the relu of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- relu(x)
    """
    s = np.maximum(0,x)
    
    return s




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
        
    a=sigmoid(z)
    activations.append(a)
    Zs.append(z)    
    return activations,Zs



def forward_prop_with_dropout(X,weights,biases,keep_prop):
    L=len(weights)
    a=X
    activations=[]
    Zs=[]
    Ds=[]
    activations.append(a)
    for l in range(L-1):
        #print(weights['w'+str(l+1)].shape)
        #print(biases['b'+str(l+1)].shape)
        #print(z.shape)
        z=np.dot(a,weights['w'+str(l+1)])+biases['b'+str(l+1)]
        
        a=relu(z)
        D = np.random.rand(a.shape[0], a.shape[1])
        D=D<keep_prop
        Ds.append(D)
        a=a*D/keep_prop
        #a=sigmoid(z)
        activations.append(a)
        Zs.append(z)
    
    z=np.dot(a,weights['w'+str(L)])+biases['b'+str(L)]
        
    a=sigmoid(z)
    activations.append(a)
    Zs.append(z)    
    return activations,Zs,Ds


def compute_loss(a, Y):
    
    """
    Implement the loss function
    
    Arguments:
    a3 -- post-activation, output of forward propagation
    Y -- "true" labels vector, same shape as a3
    
    Returns:
    loss - value of the loss function
    """
    
    m = Y.shape[0]
    logprobs = np.multiply(-np.log(a),Y) + np.multiply(-np.log(1 - a), 1 - Y)
    loss = 1./m * np.nansum(logprobs)
    
    return loss

def compute_cost_with_regularization(a, Y, weights, lambd):
    m = Y.shape[0]
    L=len(weights)
    cross_entropy_cost=compute_loss(a, Y)
   
    sum_weights=0 
    for i in range(L):
      sum_weights =sum_weights+np.sum(np.square(weights['w'+str(i+1)]))
      
    l2_regularization_cost=lambd * sum_weights/ (2 * m)
   
    cost=cross_entropy_cost+l2_regularization_cost
   
    return cost
   
def predict(X, y, weights,biases):    
    """
    This function is used to predict the results of a  n-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[0]
    p = np.zeros((m,1), dtype = np.int)
    
    # Forward propagation
    activations,Zs = forward_prop(X,weights,biases)
    
    # convert probas to 0/1 predictions
    for i in range(0, activations[-1].shape[0]):
        if activations[-1][i] > 0.5:
            p[i,0] = 1
        else:
            p[i,0] = 0

    # print results
    print("Accuracy: "  + str(np.mean((p[:,0] == y[:,0]))))
    
    return p


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



def back_prop_with_regularization(weights,activations,Zs,y,m,lambd):
    L=len(weights)

    grad={}
    dz =  (activations[-1] - y)
    dw = 1./m *np.dot(activations[-2].T,dz)+(lambd * weights['w'+str(L)]) / m
    db = 1./m *np.sum(dz, axis=0, keepdims = True)
    grad['w'+str(L)]=dw
    grad['b'+str(L)]=db
    

    for l in reversed(range(1,L)):
       da= np.dot(weights['w'+str(l+1)], dz.T)
       dz = np.multiply(da.T, np.int64(activations[l] > 0))
       #dz = 1./m *np.multiply(da.T,sigmoid_prime(Zs[i-1]))
       dw = 1./m *np.dot(activations[l-1].T,dz)+(lambd * weights['w'+str(l)]) / m
       db = 1./m *np.sum(dz, axis=0, keepdims = True)
      
       grad['w'+str(l)]=dw
       grad['b'+str(l)]=db
    
    return   grad



def back_prop_with_dropout(weights,activations,Zs,Ds,y,m):
    grad={}
    dz =  (activations[-1] - y)
    dw = 1./m *np.dot(activations[-2].T,dz)
    db = 1./m *np.sum(dz, axis=0, keepdims = True)
    L=len(weights)
    grad['w'+str(L)]=dw
    grad['b'+str(L)]=db
    
    for i in reversed(range(1,L)):
       da= np.dot(weights['w'+str(i+1)], dz.T)
       da=da*Ds[i-1].T
       dz = np.multiply(da.T, np.int64(activations[i] > 0))
       #dz = 1./m *np.multiply(da.T,sigmoid_prime(Zs[i-1]))
       dw = 1./m *np.dot(activations[i-1].T,dz)
       db = 1./m *np.sum(dz, axis=0, keepdims = True)
      
       grad['w'+str(i)]=dw
       grad['b'+str(i)]=db
      
    return   grad


def initialize_velocity(weights,biases):
    L=len(weights)
    Vdw={}
    Vdb={}
    for l in range(L):
      Vdw['dw'+str(l+1)]=np.zeros_like(weights['w'+str(l+1)])
      Vdb['db'+str(l+1)]=np.zeros_like(biases['b'+str(l+1)])
    return Vdw,Vdb            


def initialize_adam(weights,biases) :
        
    Vdw,Vdb =initialize_velocity(weights,biases)
    Sdw,Sdb =initialize_velocity(weights,biases)
    
    return  Vdw,Vdb,Sdw,Sdb

       
       
def update_params(weights,biases,grad,learning_rate):
    L=len(weights)
    for l in range(1,L):
        weights['w'+str(l+1)]=weights['w'+str(l+1)]-learning_rate*grad['w'+str(l+1)]
        biases['b'+str(l+1)]=biases['b'+str(l+1)]-learning_rate*grad['b'+str(l+1)]
    


def update_params_with_momentum(weights,biases,Vdw,Vdb,grad,beta,learning_rate):
    L=len(weights)
    for l in range(L):
       Vdw['dw'+str(l+1)]=beta*Vdw['dw'+str(l+1)]+(1-beta)*grad['w'+str(l+1)]
       Vdb['db'+str(l+1)]=beta*Vdb['db'+str(l+1)]+(1-beta)*grad['b'+str(l+1)]
    
       weights['w'+str(l+1)]=weights['w'+str(l+1)]-learning_rate*Vdw['dw'+str(l+1)]
       biases['b'+str(l+1)]=biases['b'+str(l+1)]-learning_rate*Vdb['db'+str(l+1)]


def update_parameters_with_adam(weights,biases,Vdw,Vdb,Sdw,Sdb,grads,t,learning_rate=0.01,
                                beta1=0.9, beta2=0.999, epsilon=1e-8):
    Vdw_corrected = {}                         # Initializing first moment estimate, python dictionary
    Vdb_corrected = {}   
    Sdw_corrected = {}
    Sdb_corrected = {}
    
    L=len(weights)
    for l in range(L):
        
        Vdw['dw'+str(l+1)] = beta1 * Vdw['dw'+str(l+1)] + (1 - beta1) * grads['w'+ str(l + 1)]
        Vdb['db' + str(l + 1)] = beta1 * Vdb['db' + str(l + 1)] + (1 - beta1) * grads['b' + str(l + 1)]
        Vdw_corrected['dw'+str(l+1)] = Vdw['dw'+str(l+1)] / (1 - np.power(beta1, t))
        Vdb_corrected['db' + str(l + 1)] = Vdb['db' + str(l + 1)] / (1 - np.power(beta1, t))
        Sdw['dw'+str(l+1)] = beta2 * Sdw['dw'+str(l+1)] + (1 - beta2) * np.power(grads['w' + str(l + 1)], 2)
        Sdb['db' + str(l + 1)] = beta2 * Sdb['db' + str(l + 1)] + (1 - beta2) * np.power(grads['b' + str(l + 1)], 2)
        Sdw_corrected['dw'+str(l+1)] = Sdw['dw'+str(l+1)] / (1 - np.power(beta2, t))
        Sdb_corrected['db' + str(l + 1)] = Sdb['db' + str(l + 1)] / (1 - np.power(beta2, t))
        weights['w'+ str(l + 1)] = weights['w' + str(l + 1)] - learning_rate * Vdw_corrected['dw' + str(l + 1)] / np.sqrt(Sdw_corrected['dw' + str(l + 1)] + epsilon)
        biases['b' + str(l + 1)] = biases['b' + str(l + 1)] - learning_rate * Vdb_corrected['db' + str(l + 1)] / np.sqrt(Sdb_corrected['db' + str(l + 1)] + epsilon)



def gradient_descenet(X,y,weights,biases,learning_rate,m):
    activations,Zs  =forward_prop(X,weights,biases)
        
    grad=back_prop(weights,activations,Zs,y,m)
        
    update_params(weights,biases,grad,learning_rate)

    loss=compute_loss(activations[-1], y)
    
    return loss

def gradient_descenet_with_momentum(X,y,weights,biases,Vdw,Vdb,beta,learning_rate,m):
    activations,Zs  =forward_prop(X,weights,biases)
        
    grad=back_prop(weights,activations,Zs,y,m)
        
    update_params_with_momentum(weights,biases,Vdw,Vdb,grad,beta,learning_rate)    

    loss=compute_loss(activations[-1], y)
    
    return loss

def gradient_descenet_with_adam(X,y,weights,biases,Vdw,Vdb,Sdw,Sdb,t,learning_rate,m):
    activations,Zs  =forward_prop(X,weights,biases)
        
    grad=back_prop(weights,activations,Zs,y,m)
    
    update_parameters_with_adam(weights,biases,Vdw,Vdb,Sdw,Sdb,grad,2,learning_rate=0.01,
                                beta1=0.9, beta2=0.99, epsilon=1e-8)    
    #update_params(weights,biases,grad,learning_rate)

    loss=compute_loss(activations[-1], y)
    
    return loss

def gradient_descenet_with_regularization(X,y,weights,biases,learning_rate,lambd,m):
    activations,Zs  =forward_prop(X,weights,biases)
    grad=back_prop_with_regularization(weights,activations,Zs,y,m,lambd)
        
    update_params(weights,biases,grad,learning_rate)
    loss=compute_cost_with_regularization(activations[-1], y, weights, lambd)
       
    return loss

def gradient_descenet_with_momentum_with_regularization(X,y,weights,biases,Vdw,Vdb,beta,learning_rate,lambd,m):
    activations,Zs  =forward_prop(X,weights,biases)
    grad=back_prop_with_regularization(weights,activations,Zs,y,m,lambd)
        
    update_params_with_momentum(weights,biases,Vdw,Vdb,grad,beta,learning_rate)
    loss=compute_cost_with_regularization(activations[-1], y, weights, lambd)
       
    return loss

def gradient_descenet_with_adam_with_regularization(X,y,weights,biases,Vdw,Vdb,Sdw,Sdb,t,lambd,learning_rate,m):
    activations,Zs  =forward_prop(X,weights,biases)
    grad=back_prop_with_regularization(weights,activations,Zs,y,m,lambd)
    
    #grad=back_prop(weights,activations,Zs,y,m)
    
    update_parameters_with_adam(weights,biases,Vdw,Vdb,Sdw,Sdb,grad,2,learning_rate=0.01,
                                beta1=0.9, beta2=0.99, epsilon=1e-8)    
    #update_params(weights,biases,grad,learning_rate)

    loss=compute_loss(activations[-1], y)
    
    return loss
def gradient_descenet_with_Dropout(X,y,weights,biases,learning_rate,keep_prop,m):
    activations,Zs,Ds=forward_prop_with_dropout(X,weights,biases,keep_prop)
    grad=back_prop_with_dropout(weights,activations,Zs,Ds,y,m)
        
    update_params(weights,biases,grad,learning_rate)
    loss=compute_loss(activations[-1], y)
    
    return loss

def gradient_descenet_with_momentum_with_Dropout(X,y,weights,biases,Vdw,Vdb,beta,learning_rate,keep_prop,m):
    activations,Zs,Ds=forward_prop_with_dropout(X,weights,biases,keep_prop)
    grad=back_prop_with_dropout(weights,activations,Zs,Ds,y,m)
        
    update_params_with_momentum(weights,biases,Vdw,Vdb,grad,beta,learning_rate)    
    loss=compute_loss(activations[-1], y)
    
    return loss

def gradient_descenet_with_adam_with_Dropout(X,y,weights,biases,Vdw,Vdb,Sdw,Sdb,t,keep_prop,learning_rate,m):
    activations,Zs,Ds=forward_prop_with_dropout(X,weights,biases,keep_prop)
    grad=back_prop_with_dropout(weights,activations,Zs,Ds,y,m)
    
    update_parameters_with_adam(weights,biases,Vdw,Vdb,Sdw,Sdb,grad,2,learning_rate=0.01,
                                beta1=0.9, beta2=0.99, epsilon=1e-8)    
    #update_params(weights,biases,grad,learning_rate)

    loss=compute_loss(activations[-1], y)
    
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


def mini_batch_SGD_with_regularization(mini_batches,batch_size,weights,biases,learning_rate,iteration):
    losses=[]
    t=0
    for i in range(iteration):
        c=0
        for batch in mini_batches:
            loss=gradient_descenet_with_regularization(batch[0],batch[1],weights,biases,learning_rate,lambd,batch[0].shape[0])
            losses.append(loss)
            print("epoch : %5d, batch : %5d loss : %5.5f"%(i+1,c+1,loss))
            c+=1
            t+t+1

    return losses    

def mini_batch_SGD_with_dropout(mini_batches,batch_size,weights,biases,learning_rate,iteration):
    losses=[]
    t=0
    for i in range(iteration):
        c=0
        for batch in mini_batches:
            loss=gradient_descenet_with_Dropout(batch[0],batch[1],weights,biases,learning_rate,keep_prop,batch[0].shape[0])
            losses.append(loss)
            print("epoch : %5d, batch : %5d loss : %5.5f"%(i+1,c+1,loss))
            c+=1
            t+t+1

    return losses    




def mini_batch_SGD_with_adam(mini_batches,batch_size,weights,biases,Vdw,Vdb,Sdw,Sdb,learning_rate,iteration):
    losses=[]
    t=0
    for i in range(iteration):
        c=0
        for batch in mini_batches:
            loss=gradient_descenet_with_adam(batch[0],batch[1],weights,biases,Vdw,Vdb,Sdw,Sdb,t,learning_rate,batch[0].shape[0])
            losses.append(loss)
            print("epoch : %5d, batch : %5d loss : %5.5f"%(i+1,c+1,loss))
            c+=1
            t+t+1

    return losses 


def mini_batch_SGD_with_adam_with_regularization(mini_batches,batch_size,weights,biases,Vdw,Vdb,Sdw,Sdb,lambd,learning_rate,iteration):
    losses=[]
    t=0
    for i in range(iteration):
        c=0
        for batch in mini_batches:
            loss=gradient_descenet_with_adam_with_regularization(batch[0],batch[1],weights,biases,Vdw,Vdb,Sdw,Sdb,t,lambd,learning_rate,batch[0].shape[0])
            losses.append(loss)
            print("epoch : %5d, batch : %5d loss : %5.5f"%(i+1,c+1,loss))
            c+=1
            t+t+1

    return losses 



def mini_batch_SGD_with_adam_with_dropout(mini_batches,batch_size,weights,biases,Vdw,Vdb,Sdw,Sdb,keep_prop,learning_rate,iteration):
    losses=[]
    t=0
    for i in range(iteration):
        c=0
        for batch in mini_batches:
            loss=gradient_descenet_with_adam_with_Dropout(batch[0],batch[1],weights,biases,Vdw,Vdb,Sdw,Sdb,t,keep_prop,learning_rate,batch[0].shape[0])
            losses.append(loss)
            print("epoch : %5d, batch : %5d loss : %5.5f"%(i+1,c+1,loss))
            c+=1
            t+t+1

    return losses 




def mini_batch_SGD_with_momemtum(mini_batches,batch_size,weights,biases,Vdw,Vdb,beta,learning_rate,iteration):
    losses=[]
    t=0
    for i in range(iteration):
        c=0
        for batch in mini_batches:
            loss=gradient_descenet_with_momentum(batch[0],batch[1],weights,biases,Vdw,Vdb,beta,learning_rate,batch[0].shape[0])
            losses.append(loss)
            print("epoch : %5d, batch : %5d loss : %5.5f"%(i+1,c+1,loss))
            c+=1
            t+t+1

    return losses 


def mini_batch_SGD_with_momemtum_with_regularization(mini_batches,batch_size,weights,biases,Vdw,Vdb,beta,lambd,learning_rate,iteration):
    losses=[]
    t=0
    for i in range(iteration):
        c=0
        for batch in mini_batches:
            loss=gradient_descenet_with_momentum_with_regularization(batch[0],batch[1],weights,biases,Vdw,Vdb,beta,learning_rate,lambd,batch[0].shape[0])
            losses.append(loss)
            print("epoch : %5d, batch : %5d loss : %5.5f"%(i+1,c+1,loss))
            c+=1
            t+t+1

    return losses 



def mini_batch_SGD_with_momentum_with_dropout(mini_batches,batch_size,weights,biases,Vdw,Vdb,Sdw,Sdb,keep_prop,learning_rate,iteration):
    losses=[]
    t=0
    for i in range(iteration):
        c=0
        for batch in mini_batches:
            loss=gradient_descenet_with_momentum_with_Dropout(batch[0],batch[1],weights,biases,Vdw,Vdb,Sdw,Sdb,beta,learning_rate,keep_prop,batch[0].shape[0])
            losses.append(loss)
            print("epoch : %5d, batch : %5d loss : %5.5f"%(i+1,c+1,loss))
            c+=1
            t+t+1

    return losses 








def model(X,y,network,iteration=100,optimizer='SGD',regularization='none',learning_rate=0.01,lambd=0.2,keep_prop=0.9,beta=0.9,batch_size=64 ):  
    mini_batches=random_mini_batch(X,y,batch_size,seed=0)
    weights,biases=init_weights(network)
    
    if(optimizer=='adam' and regularization=='L2'):
       Vdw,Vdb,Sdw,Sdb=initialize_adam(weights,biases)
       loss=mini_batch_SGD_with_adam_with_regularization(mini_batches,batch_size,weights,biases,Vdw,Vdb,Sdw,Sdb,lambd,learning_rate,iteration)
       
    elif(optimizer=='adam' and regularization=='dropout'):
       Vdw,Vdb,Sdw,Sdb=initialize_adam(weights,biases)
       loss=mini_batch_SGD_with_adam_with_dropout(mini_batches,batch_size,weights,biases,Vdw,Vdb,Sdw,Sdb,keep_prop,learning_rate,iteration)
    
    elif(optimizer=='adam' and regularization=='none'):
       Vdw,Vdb,Sdw,Sdb=initialize_adam(weights,biases)    
       loss=mini_batch_SGD_with_adam(mini_batches,batch_size,weights,biases,Vdw,Vdb,Sdw,Sdb,learning_rate,iteration)
    
    elif(optimizer=='momentum' and regularization=='L2'):
       Vdw,Vdb=initialize_velocity(weights,biases)
       loss=mini_batch_SGD_with_momemtum_with_regularization(mini_batches,batch_size,weights,biases,Vdw,Vdb,beta,lambd,learning_rate,iteration)
    elif(optimizer=='momentum' and regularization=='dropout'):
       Vdw,Vdb=initialize_velocity(weights,biases)
       loss=mini_batch_SGD_with_momentum_with_dropout(mini_batches,batch_size,weights,biases,Vdw,Vdb,Sdw,Sdb,keep_prop,learning_rate,iteration)
    elif(optimizer=='momentum' and regularization=='none'):
       Vdw,Vdb=initialize_velocity(weights,biases)
       loss=mini_batch_SGD_with_momemtum(mini_batches,batch_size,weights,biases,Vdw,Vdb,beta,learning_rate,iteration)
       
    
    elif(optimizer=='SGD' and regularization=='L2'):
       loss=mini_batch_SGD_with_regularization(mini_batches,batch_size,weights,biases,learning_rate,iteration)
    elif(optimizer=='SGD' and regularization=='dropout'):
       loss=mini_batch_SGD_with_dropout(mini_batches,batch_size,weights,biases,learning_rate,iteration)
    elif(optimizer=='SGD' and regularization=='none'):   
       loss=mini_batch_SGD(mini_batches,batch_size,weights,biases,learning_rate,iteration)
     
    return weights,biases,loss

network=[2,50,20,1]
m=300 
learning_rate=0.01
lambd=0.2
keep_prop=0.9
beta=0.9
iteration=100
batch_size=64  
'''
batches=random_mini_batch(X,y,batch_size,seed=0)

weights,biases=init_weights(network)
#Vdw,Vdb=initialize_velocity(weights,biases)
Vdw,Vdb,Sdw,Sdb=initialize_adam(weights,biases)
#act,Z  =forward_prop(X,weights,biases)
#loss=gradient_descenet_with_Dropout(X,y,weights,biases,learning_rate,keep_prop,m,iteration)
loss=mini_batch_SGD(batches,batch_size,weights,biases,Vdw,Vdb,Sdw,Sdb,beta,learning_rate,keep_prop,m,iteration)
predict(X, y, weights,biases)
plt.plot(loss)
# Show the plot
plt.show()

test_X=test_X.T
test_Y=test_Y.T
predict(test_X, test_Y, weights,biases)
'''





#train_X, train_Y, test_X, test_Y=load_dataset()
#X=train_X.T
#y=train_Y.T


'''
weights,biases,loss=model(X,y,network,iteration=100,optimizer='momentum',regularization='none',learning_rate=0.01,lambd=0.2,keep_prop=0.9,beta=0.9,batch_size=64 )
predict(X, y, weights,biases)
plt.plot(loss)
# Show the plot
plt.show()
test_X=test_X.T
test_Y=test_Y.T
predict(test_X, test_Y, weights,biases)
'''  