from builtins import object
import numpy as np

from ..layers import *
from ..fast_layers import *
from ..layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        #First, we unpack the input dimensions to get the data on the input
        (chanl, heigt, widt) = input_dim
        filter_height = filter_width = filter_size
        #Second, initiate Weights W1 as random values scaled appropirately and bias b1 as zeros
        self.params["W1"] = np.random.normal(scale = weight_scale, size = (num_filters, chanl, filter_height, filter_width))
        self.params["b1"] = np.zeros(num_filters)

        #Third, find out the output sizes after convolution of the hidden affine layer
        #After the first convolution layer, the size of the layers changes, so accordingly the size of weight matrices changes
        #We know the formula for padding and the output dimensions, applying them we get
        #We use the generally followed CNN architechture of zeropadding with stride = 1
        pad = (filter_size - 1)//2
        H_out_hidden = int((heigt + 2 * pad - filter_size) + 1)
        W_out_hidden = int((widt + 2 * pad - filter_size) + 1)

        #Fourth, find out the output size after maxpool of the output affine layer
        #since we know its a 2x2 max pool layer
        pool_height = pool_width = 2
        stride = 2 #commmonly used in CNNs
        H_out = int(1 + (H_out_hidden - pool_height)/stride)
        W_out = int(1 + (W_out_hidden - pool_height)/stride)

        #Fifth, we find the volume of the output size
        out_size = num_filters * H_out * W_out

        #Sixth, initiate Weights W2 as random values scaled appropirately and bias b2 as zeros for the hidden dimension
        self.params["W2"] = np.random.normal(scale = weight_scale, size = (out_size, hidden_dim))
        self.params["b2"] = np.zeros(hidden_dim)

        #Seventh, initiate Weights W3 as random values scaled appropirately and bias b3 as zeros for the final class output
        self.params["W3"] = np.random.normal(scale = weight_scale, size = (hidden_dim, num_classes))
        self.params["b3"] = np.zeros(num_classes)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        #Forward pass for a three layered CNN, using the faster implementation of the functions
        #The implementation must be as per the architecture shown below
        # conv - relu - 2x2 max pool - affine - relu - affine - softmax
        #Based on the avaiable functions in the util file we call the relevant functions in order
        #First, call the conv_relu_pool function for forward pass
        output_convolution, cache_convolution = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        #Second, call the affine_relu function for the forward pass
        output_affine_relu, cache_affine_relu = affine_relu_forward(output_convolution, W2, b2)
        #third, call the final affine layer for forward pass
        scores, cache_affine = affine_forward(output_affine_relu, W3, b3)
        #These scores are sent in the backprop to calculate the loss


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        #Calculating the losses at the output for the scores
        loss, dx = softmax_loss(scores, y) #dx is the gradient at output
        # Add regularization to the loss.
        loss += 0.5 * self.reg * (np.sum(self.params['W1'] * self.params['W1']) + np.sum(self.params['W2'] * self.params['W2']) + np.sum(self.params['W3'] * self.params['W3']))

        #Backpropagation, we need to apply back pass similarly to the layer groups as forward pass
        #First, call backpass for last affine layer
        dx3, dw3, db3 = affine_backward(dx, cache_affine)

        #Second,  call the backward for function affine relu layer
        dx2, dw2, db2 = affine_relu_backward(dx3, cache_affine_relu)

        #Second, call backpass for conv_relu_pool layer 1
        dx3, dw1, db1 = conv_relu_pool_backward(dx2, cache_convolution)
        
        #Third, regularize the dWs for all the weight matrices
        dw1 += self.reg * self.params['W1']
        dw2 += self.reg * self.params['W2']
        dw3 += self.reg * self.params['W3']

        #Last, save the gradients for each in the grads dict.
        grads['W1'] = dw1
        grads['b1'] = db1
        grads['W2'] = dw2
        grads['b2'] = db2
        grads['W3'] = dw3
        grads['b3'] = db3
        #that ends the backpropogation


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
