from builtins import range
import numpy as np
from random import shuffle
from past.builtins import *
import math


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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)  
        #First, we shift the values so that the highest number is zero, so that it doesn't cause numeric destability
        scores = scores - scores.max()
        #Second, find the correct class score and estimate its exponential
        correct_class_score = scores[y[i]]
        exp_correct_score = math.exp(correct_class_score)
        #Third, take exponential of all the scores and sum them up
        exp_scores = np.exp(scores)
        summed_exp_scores = np.sum(exp_scores)
        exp_ratio = exp_correct_score/summed_exp_scores
        #Fourth, input them all into the softmax loss formula and calculate the loss
        loss_value = -1 * np.log(exp_ratio)
        loss += loss_value #since we need sum over all datapoints, we keep updating the loss value

        #calculate the gradient dW
        #First, for the correct class through the differential formula
        dW[:, y[i]] = dW[:, y[i]] -1 * (summed_exp_scores - exp_correct_score) / summed_exp_scores * X[i]
        #Second, for all the other incorrect classes
        for j in range(num_classes):
          if j == y[i]:
            continue
          dW[:, j] = dW[:, j] + np.exp(scores[j])/summed_exp_scores * X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #take average of the dW over all samples
    dW /= num_train
    #regularize the partial derivative
    dW += reg * 2 * W # if we diff W^2 we get 2W, apply that to the regularization term

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1]
    num_train = X.shape[0]
    scores = X.dot(W)
    #First, we shift the values so that the highest number is zero, so that it doesn't cause numeric destability
    scores = scores - scores.max()
    #Second, find the correct class score and estimate its exponential over all training data
    correct_class_score = scores[range(0, num_train), y]
    exp_correct_score = np.exp(correct_class_score)
    #Third, take exponential of all the scores and sum them up
    exp_scores = np.exp(scores)
    summed_exp_scores = np.sum(exp_scores, axis = 1)
    exp_ratio = exp_correct_score/summed_exp_scores
    #Fourth, input them all into the softmax loss formula and calculate the loss
    loss_value = np.log(exp_ratio)
    loss = -1 * np.sum(loss_value) #since we need sum over all datapoints, we add them all up

    #calculate the gradient dW
    #First, we normalize all the scores
    gradient_ratio = (exp_scores) / (summed_exp_scores.reshape(num_train, 1))
    #Second, we then estimate how much it deviates
    gradient_ratio[range(0, num_train), y] = -1 *  (summed_exp_scores - exp_correct_score)/ summed_exp_scores
    dW = X.T.dot(gradient_ratio)


    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #take average of the dW over all samples
    dW /= num_train
    #regularize the partial derivative
    dW += reg * 2 * W # if we diff W^2 we get 2W, apply that to the regularization term

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
