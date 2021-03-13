#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import deep_neural_net as dn
import h5py

get_ipython().run_line_magic('matplotlib', 'inline')


# In[17]:


def load_data():
    train_dataset = h5py.File('train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


# In[18]:


train_x_orig, train_y, test_x_orig, test_y, classes = load_data()


# In[19]:


# Explore your dataset 
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))


# In[20]:


# Reshape the training and test examples 
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten / 255.
test_x = test_x_flatten / 255.

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))


# In[36]:



import numpy as np
import matplotlib.pyplot as plt
import h5py


def sigmoid(Z):
  A = 1/(1+np.exp(-Z))
  cache = Z
  
  return A, cache

def relu(Z):
  A = np.maximum(0,Z)
  cache = Z 
  return A, cache

def tanh(Z):
  A = (np.exp(Z) - np.exp(-1.0*Z))/(np.exp(Z) + np.exp(-1.0*Z))
  cache = Z
  return A,Z

def leaky_relu(Z):
  A  = np.maximum(0.01 * Z, Z)
  cache = Z
  return A,Z

def relu_backward(dA, cache):
  Z = cache
  dZ = np.array(dA, copy=True) 
  dZ[Z <= 0] = 0
  return dZ

def sigmoid_backward(dA, cache):
  Z = cache
  s = 1/(1+np.exp(-Z))
  dZ = dA * s * (1-s)
  return dZ

def tanh_backword(dA,cache):
  Z = cache
  s = (np.exp(Z) - np.exp(-1.0*Z))/(np.exp(Z) + np.exp(-1.0*Z))
  dZ = dA * (1-s**2)
  return dZ

def leaky_relu_backward(dA, cache):
  Z = cache
  dZ = np.array(dA, copy=True) 
  dZ[Z <= 0] = 0.01
  return dZ    


def initialize_parameters_deep(layer_dims):
  np.random.seed(1)
  parameters = {}
  L = len(layer_dims)            # number of layers in the network

  for l in range(1, L):
      parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) #*0.01
      parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
      
  return parameters

def linear_forward(A, W, b):
  Z = W.dot(A) + b
  cache = (A, W, b)
  return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
  if activation == "sigmoid":
      Z, linear_cache = linear_forward(A_prev, W, b)
      A, activation_cache = sigmoid(Z)
      
  elif activation == "relu":
      Z, linear_cache = linear_forward(A_prev, W, b)
      A, activation_cache = relu(Z)
      
  elif activation == "tanh":
      Z, linear_cache = linear_forward(A_prev, W, b)
      A, activation_cache = tanh(Z)
      
  elif activation == "leaky_relu":
      Z, linear_cache = linear_forward(A_prev, W, b)
      A, activation_cache = leaky_relu(Z)
  
  cache = (linear_cache, activation_cache)

  return A, cache

def L_model_forward(X, parameters):
  caches = []
  A = X
  L = len(parameters) // 2
  
  for l in range(1, L):
      A_prev = A 
      A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
      caches.append(cache)
  
  AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")
  caches.append(cache)
  
  return AL, caches

def compute_cost(AL, Y):
  m = Y.shape[1]
  cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
  
  cost = np.squeeze(cost)     
  return cost

def linear_backward(dZ, cache):
  A_prev, W, b = cache
  m = A_prev.shape[1]

  dW = 1./m * np.dot(dZ,A_prev.T)
  db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
  dA_prev = np.dot(W.T,dZ)
  
  return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
  linear_cache, activation_cache = cache
  
  if activation == "relu":
      dZ = relu_backward(dA, activation_cache)
      dA_prev, dW, db = linear_backward(dZ, linear_cache)
      
  elif activation == "sigmoid":
      dZ = sigmoid_backward(dA, activation_cache)
      dA_prev, dW, db = linear_backward(dZ, linear_cache)
   
  elif activation == "tanh":
      dZ = tanh_backward(dA, activation_cache)
      dA_prev, dW, db = linear_backward(dZ, linear_cache)
      
  elif activation == "leaky_relu":
      dZ = leaky_relu_backward(dA, activation_cache)
      dA_prev, dW, db = linear_backward(dZ, linear_cache)    
  
  return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
  grads = {}
  L = len(caches) 
  m = AL.shape[1]
  Y = Y.reshape(AL.shape) 
  
  dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
  
  current_cache = caches[L-1]
  grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")
  
  for l in reversed(range(L-1)):
      current_cache = caches[l]
      dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation = "relu")
      grads["dA" + str(l + 1)] = dA_prev_temp
      grads["dW" + str(l + 1)] = dW_temp
      grads["db" + str(l + 1)] = db_temp

  return grads

def update_parameters(parameters, grads, learning_rate):
  L = len(parameters) // 2 
  
  for l in range(L):
      parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
      parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
      
  return parameters

def predict(X, y, parameters):
  
  m = X.shape[1]
  n = len(parameters) // 2
  p = np.zeros((1,m))
  
  
  probas, caches = L_model_forward(X, parameters)

  for i in range(0, probas.shape[1]):
      if probas[0,i] > 0.5:
          p[0,i] = 1
      else:
          p[0,i] = 0
          
  print("Accuracy: "  + str(np.sum((p == y)/m)))
      
  return p

def print_mislabeled_images(classes, X, y, p):
  a = p + y
  mislabeled_indices = np.asarray(np.where(a == 1))
  plt.rcParams['figure.figsize'] = (40.0, 40.0) # set default size of plots
  num_images = len(mislabeled_indices[0])
  for i in range(num_images):
      index = mislabeled_indices[1][i]
      
      plt.subplot(2, num_images, i + 1)
      plt.imshow(X[:,index].reshape(64,64,3), interpolation='nearest')
      plt.axis('off')
      plt.title("Prediction: " + classes[int(p[0,index])].decode("utf-8") + " \n Class: " + classes[y[0,index]].decode("utf-8"))


# In[37]:


def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False): #lr was 0.009
    np.random.seed(1)
    costs = []                      
    parameters = initialize_parameters_deep(layers_dims)
    
    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters


# In[55]:


### CONSTANTS ###
layers_dims = [12288, 20, 10, 7, 1] #  5-layer model
parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations=2500, print_cost=True)


# In[56]:


pred_train = predict(train_x, train_y, parameters)


# In[57]:


pred_test = predict(test_x, test_y, parameters)


# In[ ]:




