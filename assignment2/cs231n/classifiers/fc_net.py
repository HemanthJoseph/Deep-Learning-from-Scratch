from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        #First, initiate Weights W1 and W2 as random values scaled appropirately
        #Second, initiate biases b1 and b2 as zeros
        #To generalize our initialization for many layers, we need to make a data structure to 
        #hold all the data for the different layers
        layers_data = np.hstack([input_dim, hidden_dims, num_classes]) #helps for easy mapping between many layers
        for i in range(self.num_layers):
          self.params['W' + str(i+1)] = weight_scale * np.random.randn(layers_data[i], layers_data[i+1])
          self.params['b' + str(i+1)] = np.zeros(layers_data[i+1])

        # We need to use this to perform batch normalization
        #Check if the flag for normalization is not set to None
        if (self.normalization != None):
          for i in range(self.num_layers-1):
            #Initialize gamma as ones and beta as zero
            self.params['gamma' + str(i+1)] = np.ones(layers_data[i+1])
            self.params['beta' + str(i+1)] = np.zeros(layers_data[i+1])

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        #Forward pass
        #To generalize, we make certain common data variable to hold all the cache data to be accessed during back-propogation
        cache_affine = []
        cache_relu = []
        cache_batch_norm = []
        cache_dropout = [] #To store the cache data
        forward_x = X #so that X doesn't get updated each time we call it
        #Next we need to iterate through all the L-1 layers to perform forward propogation and then perform the forward affine for the last layer separately
        for i in range(self.num_layers-1):
            #First, call affine forward for current layer
            affine_current_layer_out, cache = affine_forward(forward_x, self.params['W' + str(i+1)], self.params['b' + str(i+1)])
            cache_affine.append(cache)
            
            #Second, we apply a batch normalization befor we do the ReLU for this layer
            if (self.normalization != None):
              batchnorm_current_layer_out, cache = batchnorm_forward(affine_current_layer_out, self.params['gamma' + str(i+1)], self.params['beta' + str(i+1)], self.bn_params[i])
              cache_batch_norm.append(cache)
            else: #I am using this because I was getting an error in solver due to variable names (condition where we don't use batchnorm)
              batchnorm_current_layer_out = affine_current_layer_out
            #Third, call relu forward for current layer
            relu_out, cache = relu_forward(batchnorm_current_layer_out)
            cache_relu.append(cache)
            #Fourth we call the dropout layer
            if (self.use_dropout):
              forward_x, cache = dropout_forward(relu_out, self.dropout_param)
              cache_dropout.append(cache)
            else:
              forward_x = relu_out
        #Fifth, call affine forward for the last layer
        affine_layer_last, cache = affine_forward(forward_x, self.params['W' + str(self.num_layers)], self.params['b' + str(self.num_layers)])
        cache_affine.append(cache)
        #Store the values in scores variable as asked 
        scores = affine_layer_last

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        
        #Backpropagation
        #Calculating the losses at the output for the scores
        loss, dx = softmax_loss(scores, y) #dx is the gradient at output
        #Add regularization to the loss to the last layer
        loss += 0.5 * self.reg * (np.sum(self.params['W' + str(self.num_layers)] * self.params['W' + str(self.num_layers)]))
        #As per the architecture, similar to the forward pass, we do the backward affine for the last layer
        dx_updated, dw, db = affine_backward(dx, cache_affine[-1])
        grads['W'+ str(self.num_layers)], grads['b'+ str(self.num_layers)] = dw, db #storing them in the grads dictionary
        #Regularize the weights for the last layer
        grads['W'+ str(self.num_layers)] += self.reg * self.params['W' + str(self.num_layers)]

        #Now we continue the backpropogation for the remainining layers after the last layer
        dx_backward = dx_updated #so that we can send it back for each iteration
        for i in range(self.num_layers-1, 0, -1): #iterate from the end
          #First, we applying the dropout for the backprop
          if (self.use_dropout):
            ddrop_back = dropout_backward(dx_backward, cache_dropout[i-1])
          else:
            ddrop_back = dx_backward
          #Second,  call relu backward on current layer
          drelu_back = relu_backward(ddrop_back, cache_relu[i-1])
          #Third we perform the batchnorm for the back prop after the ReLU
          if (self.normalization != None):
            dbatchnorm_back, dgamma, dbeta = batchnorm_backward(drelu_back, cache_batch_norm[i-1])
            grads['gamma'+ str(i)], grads['beta'+ str(i)] = dgamma, dbeta #Storing them in the grads dictionary
          else: #I am using this because I was getting an error in solver due to variable names (condition where we don't use batchnorm)
              dbatchnorm_back = drelu_back
          #Fourth, call backpass for affine for current layer
          dx_backward, dw2, db2 = affine_backward(dbatchnorm_back, cache_affine[i-1])
          grads['W'+ str(i)], grads['b'+ str(i)] = dw2, db2 #storing them in the grads dictionary
          #Regularize the weights for the last layer
          grads['W'+ str(i)] += self.reg * self.params['W' + str(i)]
          #Last, we need to update the loss for this layer
          loss += 0.5 * self.reg * (np.sum(self.params['W' + str(i)] * self.params['W' + str(i)]))

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
