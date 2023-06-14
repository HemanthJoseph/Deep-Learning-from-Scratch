from builtins import range
import numpy as np



def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #First, we reshape the input to the required dimension
    #Second, multiply the inputs with the weights
    #Last, add the bias term to it
    x_reshaped = np.reshape(x, (x.shape[0], -1))
    out_prod = np.dot(x_reshaped, w)
    out = out_prod + b

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #First, we do the gradient for the added bias term, this will have the same gradient as the upstream gradient
    db=np.sum(dout,axis=0) #summing so as to collapse all the values to conform the dimensions 
    
    #The upstream gradient is split equally to the product of x and w
    #Second, we need to calculate the gradients at w
    #To get the gradient at w, dw, we need to multiply the reshaped x with the upstream gradient
    x_reshaped = np.reshape(x, (x.shape[0], -1))
    dw = x_reshaped.T @ dout

    
    #Last, we need to calculate the gradients at x
    #To get the gradient at x, dx, we need to multiply the w with the upstream gradient
    dx = w @ dout.T
    dx = dx.T
    dx = np.reshape(dx, x.shape) #reshaping it to get the required dimensions

    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # out = np.zeros(x.shape) #slower implementation
    # for i in range(x.shape[0]):
    #   for j in range(x.shape[1]):
    #     if x[i][j] < 0:
    #       out[i][j] = 0
    #     else:
    #       out[i][j] = x[i][j]

    #We check for all inputs
    #If input it positive, we return the input
    #If less than zero, we return zero
    relu = lambda x : x * (x > 0) #faster implementation
    out = relu(x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #The gradient at the input, dx is nothing but the upstream gradient multiplied by the max of the input
    #input value if input id greater than zero
    #zero if the input is less than zero
    dx = dout * (x>0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None
    ###########################################################################
    # TODO: Implement loss and gradient for multiclass SVM classification.    #
    # This will be similar to the svm loss vectorized implementation in       #
    # cs231n/classifiers/linear_svm.py.                                       #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = x.shape[0]
    #First, take the correct class scores over all the training data
    correct_class_scores = x[range(0, num_train), y]
    correct_class_scores = correct_class_scores.reshape(num_train, -1)#reshape to match dimensions
    #Second, calculate the margins for the SVM
    margin = x - correct_class_scores + 1
    #Third, set the margin for the correct class back at zero
    margin[range(0, num_train), y] = 0
    #Fourth, select the max between 0 and the margin
    individual_loss = np.maximum(0, margin)
    #Sum the losses over all training data
    loss = np.sum(individual_loss)
    #Average over all training data
    loss /= num_train

    #First, we create a mask over all our individual loss values, the same size as our margins
    #whichever value has a +ve margin set it as an 1
    mask = np.zeros(individual_loss.shape)
    mask[individual_loss > 0] = 1
    #Second, we set the correct class with the -ve values
    mask[np.arange(0, num_train), y] = -1 * np.sum(mask, axis=1) 
    dx = mask/num_train #divide by the number to get the gradient
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None
    ###########################################################################
    # TODO: Implement the loss and gradient for softmax classification. This  #
    # will be similar to the softmax loss vectorized implementation in        #
    # cs231n/classifiers/softmax.py.                                          #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = x.shape[1]
    num_train = x.shape[0]
    scores = x
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
    loss /= num_train

    #calculate the gradient dx
    #First, we normalize all the scores
    gradient_ratio = (exp_scores) / (summed_exp_scores.reshape(num_train, 1))
    #Second, we then estimate how much it deviates
    gradient_ratio[range(0, num_train), y] = -1 *  (summed_exp_scores - exp_correct_score)/ summed_exp_scores
    dx = gradient_ratio/num_train #divide by the number to get the gradient

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx
