import numpy as np
from keras.utils import to_categorical
from random import uniform

def gradCheck(inputs, target,a_prev,parameters):
  Waa, Wax, Way,ba,by = parameters['Waa'], parameters['Wxa'], parameters['Way'], parameters['ba'], parameters['by']
  num_checks, delta = 10, 1e-5
  _, gradients, _ = optimize(inputs,target, a_prev, parameters, learning_rate = 0.1)
  dWaa, dWax, dWya, dba, dby = gradients['dWaa'], gradients['dWxa'], gradients['dWay'], gradients['dba'], gradients['dby']

  for param,dparam,name in zip([Wax,Waa , Way,ba,by], [dWax,dWaa, dWya, dba, dby], ['Wxa', 'Waa', 'Way', 'ba', 'by']):
    s0 = dparam.shape
    s1 = param.shape
    #assert (s0 == s1), 'Error dims dont match: %s and %s.' % (`s0`, `s1`)
    print (name)
    for i in range(num_checks):
      ri = int(uniform(0,param.size))
      # evaluate cost at [x + delta] and [x - delta]
      old_val = param.flat[ri]
      param.flat[ri] = old_val + delta
      cg0, _, _ = optimize(inputs,target, a_prev, parameters, learning_rate = 0.1)
      param.flat[ri] = old_val - delta
      cg1, _, _ = optimize(inputs,target, a_prev, parameters, learning_rate = 0.1)
      param.flat[ri] = old_val # reset old value for this parameter
      # fetch both numerical and analytic gradient
      grad_analytic = dparam.flat[ri]
      grad_numerical = (cg0 - cg1) / ( 2 * delta )
      #rel_error = abs(grad_analytic - grad_numerical) / abs(grad_numerical + grad_analytic)
      
      numerator = np.linalg.norm(grad_analytic - grad_numerical)                                     # Step 1'
      denominator = np.linalg.norm(grad_analytic) + np.linalg.norm(grad_numerical)                   # Step 2'
      difference = numerator / denominator 
      
      
      print ('%f, %f => %e ' % (grad_numerical, grad_analytic, difference))
      # rel_error should be on order of 1e-7 or less





def generate_data_sequence(X,y,seq_len):
    num_sequence=len(X)//seq_len
    for i in range(num_sequence):
        yield X[i*seq_len:(i+1)*(seq_len)],y[i*seq_len:(i+1)*(seq_len)]


def get_batches(X,y,seq_len,):
    X_batches=[]
    y_batches=[]
    for X_,y_ in  generate_data_sequence(X,y,seq_len):
    
        X_batches.append(X_)
        y_batches.append(y_)
 
    X_batches_np=np.array(X_batches)
    y_batches_np=np.array(y_batches)
    
    X_hot_batches=to_categorical(X_batches_np)
    y_hot_batches=to_categorical(y_batches_np)
    
    
    return X_hot_batches,y_hot_batches    

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1)

def initialize_parameters(n_a, n_x, n_y):

    np.random.seed(1)
    Wxa = np.random.randn(n_x,n_a)*0.01 # input to hidden
    Waa = np.random.randn(n_a, n_a)*0.01 # hidden to hidden
    Way = np.random.randn(n_a,n_y)*0.01 # hidden to output
    ba = np.zeros((1,n_a)) # hidden bias
    by = np.zeros((1,n_y)) # output bias
    
    parameters = {"Wxa": Wxa, "Waa": Waa, "Way": Way, "ba": ba,"by": by}
    
    return parameters

def clip(gradients, maxValue):

    
    dWaa, dWax, dWya, dba, dby = gradients['dWaa'], gradients['dWxa'], gradients['dWay'], gradients['dba'], gradients['dby']
   
    ### START CODE HERE ###
    # clip to mitigate exploding gradients, loop over [dWax, dWaa, dWya, db, dby]. (≈2 lines)
    for gradient in [dWax, dWaa, dWya, dba, dby]:
        np.clip(gradient, -maxValue, maxValue, out=gradient)
    ### END CODE HERE ###
    
    gradients = {"dWaa": dWaa, "dWxa": dWax, "dWay": dWya, "dba": dba, "dby": dby}
    
    return gradients

def rnn_step_forward(xt, a_prev, parameters):

    
    # Retrieve parameters from "parameters"
    Wxa = parameters["Wxa"]
    Waa = parameters["Waa"]
    Way = parameters["Way"]
    ba = parameters["ba"]
    by = parameters["by"]
    
    ### START CODE HERE ### (≈2 lines)
    # compute next activation state using the formula given above
    
    
    a_next = np.tanh(np.dot(a_prev,Waa) + np.dot(xt,Wxa) + ba)
    # compute output of the current cell using the formula given above
    yt_pred = softmax(np.dot(a_next,Way ) + by)
    ### END CODE HERE ###
    
    # store values you need for backward propagation in cache
   
    
    return a_next, yt_pred

def rnn_step_backward(dy, gradients, parameters, x, a, a_prev):
    #print(parameters['Way'].T.shape)
    #print(dy.shape)
    #print(gradients['da_next'].shape)
    
    gradients['dWay'] += np.dot(a.T,dy)
    gradients['dby'] += dy
    da = np.dot(dy,parameters['Way'].T,) + gradients['da_next'] # backprop into h
    daraw = (1 - a * a) * da # backprop through tanh nonlinearity

    x=x.reshape([1,x.shape[0]])
    #print(daraw.shape)
    #print(parameters['Waa'].T.shape)
    gradients['dba'] += daraw
    gradients['dWxa'] += np.dot(x.T,daraw)
    gradients['dWaa'] += np.dot(a_prev.T,daraw)
    gradients['da_next'] = np.dot(daraw,parameters['Waa'].T)
    return gradients

def rnn_forward(X,y,a0,parameters):
    #print(X.shape)
    #print(y.shape)
    Y_hat={}
    a={}
    a[-1]=np.copy(a0)
    loss=0
    for t in range(X.shape[0]):
       a[t],Y_hat[t]=rnn_step_forward(X[t][:], a[t-1], parameters)
       #print(np.argmax(y[t,:]))
       #print(Y_[t][0].shape)
       loss -=np.log(Y_hat[t][0][np.argmax(y[t,:])])
 
    return loss,Y_hat,a

def rnn_backward(X, Y, parameters,  y_hat, a,seq_len):
    # Initialize gradients as an empty dictionary
    gradients = {}
    
    # Retrieve from cache and parameters
    #(y_hat, a, x) = cache
    Waa, Wax, Way, by, ba = parameters['Waa'], parameters['Wxa'], parameters['Way'], parameters['by'], parameters['ba']
    
    # each one should be initialized to zeros of the same dimension as its corresponding parameter
    gradients['dWxa'], gradients['dWaa'], gradients['dWay'] = np.zeros_like(Wax), np.zeros_like(Waa), np.zeros_like(Way)
    gradients['dba'], gradients['dby'] = np.zeros_like(ba), np.zeros_like(by)
    gradients['da_next'] = np.zeros_like(a[0])
    
    ### START CODE HERE ###
    # Backpropagate through time
    for t in reversed(range(seq_len)):
        #dy = np.copy(y_hat[t])
        #dy[Y[t]] -= 1
        dy=y_hat[t]-Y[t]
        gradients = rnn_step_backward(dy, gradients, parameters, X[t][:], a[t], a[t-1])
    ### END CODE HERE ###
    
    return gradients, a

def update_parameters(parameters, gradients, lr):

    parameters['Wxa'] += -lr * gradients['dWxa']
    parameters['Waa'] += -lr * gradients['dWaa']
    parameters['Way'] += -lr * gradients['dWay']
    parameters['ba']  += -lr * gradients['dba']
    parameters['by']  += -lr * gradients['dby']
    return parameters



def sample(parameters,a_prev, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((1,vocab_size))
  x[0][seed_ix] = 1
  ixes = []
  Wxa = parameters["Wxa"]
  Waa = parameters["Waa"]
  Way = parameters["Way"]
  ba = parameters["ba"]
  by = parameters["by"]
  #print(x.shape)  
    
    
    

  for t in range(n):
    a_next = np.tanh(np.dot(a_prev,Waa) + np.dot(x,Wxa) + ba)
    # compute output of the current cell using the formula given above
    yt_pred = softmax(np.dot(a_next,Way ) + by)
    ix = np.random.choice(range(vocab_size), p=yt_pred.ravel())

    x = np.zeros((1,vocab_size))
    x[0][ix] = 1
    ixes.append(ix)
  return ixes

def get_initial_loss(vocab_size, seq_length):
    return -np.log(1.0/vocab_size)*seq_length

def smooth(loss, cur_loss):
    return loss * 0.999 + cur_loss * 0.001

def optimize(X, Y, a_prev, parameters, learning_rate = 0.1):

    
    ### START CODE HERE ###
    
    # Forward propagate through time (≈1 line)
    loss,Y_hat,a = rnn_forward(X, Y, a_prev, parameters)
    
    # Backpropagate through time (≈1 line)
    gradients, a = rnn_backward(X, Y, parameters, Y_hat, a,seq_len)
    
    # Clip your gradients between -5 (min) and 5 (max) (≈1 line)
    gradients = clip(gradients, maxValue=5)
    
    # Update parameters (≈1 line)
    parameters = update_parameters(parameters, gradients, learning_rate)
    
    ### END CODE HERE ###
    
    return loss, gradients, a[-1]



def model(X_hot_batches,y_hot_batches,vocab_size,seq_len,hidden_len,iterations):
    parameters=initialize_parameters(hidden_len,vocab_size, vocab_size)
    a_prev = np.zeros([1,hidden_len])
    loss= get_initial_loss(vocab_size, seq_len)
    batch_size=X_hot_batches.shape[0]

    for itr in range(iterations):
       for i in range(batch_size):
           cur_loss, grad,a_prev=optimize(X_hot_batches[i,:,:], y_hot_batches[i,:,:], a_prev, parameters, learning_rate = 0.01)
           loss = smooth(loss, cur_loss)
           if i%100==0:
              sample_ix = sample(parameters,a_prev, np.argmax(X_hot_batches[i,:,:][0]), 200)
              txt = ''.join(ix_to_char[ix] for ix in sample_ix)
              print ('----\n %s \n----' % (txt, ))
              print("loss = ",loss) 
              print("iteration = ",itr) 
              

    
# data I/O
data = open('input.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print ('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

X=[char_to_ix[char]  for char in data]
y=[char_to_ix[char]  for char in data[1:]]+[' ']              
char_len=len(chars)
seq_len=50
hidden_len=100

    
X_hot_batches,y_hot_batches=get_batches(X,y,seq_len)              
model(X_hot_batches,y_hot_batches,vocab_size,seq_len,hidden_len,iterations=1)

#a_prev = np.zeros([1,hidden_len])
#parameters=initialize_parameters(hidden_len,vocab_size, vocab_size)

#gradCheck(X_hot_batches[0,:,:], y_hot_batches[0,:,:],a_prev,parameters)