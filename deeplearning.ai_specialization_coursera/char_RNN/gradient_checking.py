import numpy as np
def dirctory_to_vector(weights,biases):
    L=len(weights)
    
    for l in range(L):
        w=weights['w'+str(l+1)]
        b=biases['b'+str(l+1)]
        w=w.flatten()
        b=b.flatten()
        if(l==0):
            theta=w
            theta = np.concatenate((theta, b), axis=0)
            
        else:
            theta = np.concatenate((theta, w), axis=0)
            theta = np.concatenate((theta, b), axis=0)
 
    return theta

def vector_to_directory(theta,network):
    weights={}
    biases={}
    L=len(network)
    count=0
    for l in range(L-1):
        weights['w'+str(l+1)]=theta[count:count+(network[l]*network[l+1])].reshape([network[l],network[l+1]])
        biases['b'+str(l+1)]=theta[count+(network[l]*network[l+1]):count+(network[l]*network[l+1])+network[l+1]].reshape([1,network[l+1]])
        count=count+(network[l]*network[l+1])+network[l+1]
    
    return weights,biases

def gradients_to_vector(gradients,network):
    L=len(network)-1
    
    for l in range(L):
        w=gradients['w'+str(l+1)]
        b=gradients['b'+str(l+1)]
        w=w.flatten()
        b=b.flatten()
        if(l==0):
            theta=w
            theta = np.concatenate((theta, b), axis=0)
            
        else:
            theta = np.concatenate((theta, w), axis=0)
            theta = np.concatenate((theta, b), axis=0)
 
    return theta


def gradient_checking(weights,biases,network, gradients, X, y,forward_prop,compute_loss,epsilon=1e-7):
    parameters=dirctory_to_vector(weights,biases)
    grad=gradients_to_vector(gradients,network)
    num_parameters=grad.shape[0]
    J_plus=np.zeros(num_parameters)
    J_minus=np.zeros(num_parameters)
    grad_approx=np.zeros(num_parameters)
    
    for i in range(num_parameters):
        parameters_val_plus=np.copy(parameters)
        parameters_val_plus[i]=parameters_val_plus[i]+epsilon
        weights,biases=vector_to_directory(parameters_val_plus,network)
        activations,Zs  =forward_prop(X,weights,biases)
        loss=compute_loss(activations[-1], y)
        J_plus[i]=loss
        
        parameters_val_minus=np.copy(parameters)
        parameters_val_minus[i]=parameters_val_minus[i]-epsilon
        weights,biases=vector_to_directory(parameters_val_minus,network)
        activations,Zs  =forward_prop(X,weights,biases)
        loss=compute_loss(activations[-1], y)
        J_minus[i]=loss
        
        grad_approx[i]=(J_plus[i]-J_minus[i])/(2*epsilon)
        
    numerator = np.linalg.norm(grad - grad_approx)                                     # Step 1'
    denominator = np.linalg.norm(grad) + np.linalg.norm(grad_approx)                   # Step 2'
    difference = numerator / denominator      
    
    if difference > 1e-7:
        print("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
    else:
        print("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")
    
    return difference         

