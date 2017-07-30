import numpy as np
from random import shuffle
from past.builtins import xrange
from math import exp, log
def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  num_train = X.shape[0]
  num_class = W.shape[1]
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
 
  f = X.dot(W) #SHAPE: N x C
  df= np.zeros_like(f)
  #normalize
  #f -= np.max(f)
  print(f[0].shape)
  for i in range(num_train):
      #for j in range (num_class):
      f[i] += -np.max(f[i])
      p = exp(f[i,y[i]])/(np.sum(np.exp(f[i])))
     # print(f[i,y[i]],np.sum(f[i,:])
      loss += -np.log(p)
      for k in range(num_class):
          df[i,k] = exp(f[i,k])/(np.sum(np.exp(f[i])))
      df[i,y[i]] += -1 
  df /= num_train
      #df[]
  dW = X.T.dot(df)
  loss /= num_train
  #dW
  loss += 0.5*reg * np.sum(W * W)
  dW += reg*W
           
           
  
  #for i in range()
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  #pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  num_train = X.shape[0]
  num_class = W.shape[1]
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  f = X.dot(W) #get score X[NxD] dot W[DxC] = f[NxC]
  df= np.zeros_like(f)
  f_max =np.max(f,axis =1)
  f_norm = f - f_max[:,np.newaxis]

  p= np.zeros_like(f)
  #x = np.divide(np.exp(f_norm[range(num_train),y]),np.sum(np.exp(f_norm), axis = 1))
  #loss = np.sum(-np.log(x))
  #p_T =  np.divide(np.exp(f_norm).T,np.sum(np.exp(f_norm), axis = 1))
  #p= p_T.T
  sum_f = np.sum(np.exp(f_norm), axis=1, keepdims=True)
  p = np.exp(f_norm)/sum_f
  #loss = np.sum(-np.log(p[range(num_train),y]))
  loss = np.sum(-np.log(p[np.arange(num_train), y]))
  loss /= num_train
  loss += 0.5*reg * np.sum(W * W)
    
  df = p
  df[range(num_train),y] =p[range(num_train),y] + -1  
  #print(p.shape)
  #print('df shape', df.shape)
  
  dW = X.T.dot(df)     
  dW /= num_train
  #print(dW.shape)
  dW += reg*W
           
  #pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

