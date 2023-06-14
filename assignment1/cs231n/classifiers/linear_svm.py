from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
              loss += margin
              #Next, we do the update for the current class.
              dW[:, y[i]] = dW[:, y[i]] - X[i] #We subtract because its of the correct class. 
              dW[:,j] = dW[:,j] + X[i] #We add the values of training data because its an incorrect class

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #take average of the dW over all samples
    dW /= num_train
    #regularize the partial derivative
    dW += reg * 2 * W # if we diff ^W2 we get 2W, apply that to the regularization term

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****      

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    all_scores = X.dot(W)
    #First take the correct class scores over all the training data
    correct_class_scores = all_scores[range(0, num_train), y]
    correct_class_scores = correct_class_scores.reshape(num_train, -1)#reshape to match dimensions
    #Second, calculate the margins for the SVM
    margin = all_scores - correct_class_scores + 1
    #Third, set the margin for the correct class back at zero
    margin[range(0, num_train), y] = 0
    #Fourth, select the max between 0 and the margin
    individual_loss = np.maximum(0, margin)
    #Sum the losses over all training data
    loss = np.sum(individual_loss)
    #Average over all training data
    loss /= num_train
    #Regularize the loss
    loss += reg * np.sum(W * W)
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #Now we make a vectorized version of the gradient dW computation
    #First we create a mask over all our individual loss values, the same size as our margins
    #whichever value has a +ve margin set it as an 1
    mask = np.zeros(individual_loss.shape)
    mask[individual_loss > 0] = 1
    #Second we set the correct class with the -ve values
    mask[np.arange(0, num_train), y] = -1 * np.sum(mask, axis=1) 
    #Third, dW is calculated as the product between X transpose and the mask we just created
    dW = X.T.dot(mask)
    #Next average the gradient over all training examples
    dW /= num_train
    #regularize the partial derivative
    dW += reg * 2 * W # if we diff W^2 we get 2W, apply that to the regularization term





    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
