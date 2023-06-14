from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

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
    # TODO: Copy over your solution from Assignment 1.                        #
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
    """Computes the backward pass for an affine (fully connected) layer.

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
    # TODO: Copy over your solution from Assignment 1.                        #
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
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

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
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
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


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

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
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = x.shape[1]
    num_train = x.shape[0]
    scores = x
    #First, we shift the values so that the highest number is zero, so that it doesn't cause numeric destability
    scores = scores - scores.max(axis = 1, keepdims=True)
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


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)
    

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        #First, we take the mean for the current batch
        curr_mean = x.mean(axis = 0)
        #Second, we take the variance for the current batch, add the epsilon to avoid numeric instability
        #and then we take the standard deviation
        curr_variance = x.var(axis = 0)
        curr_std = np.sqrt(curr_variance + eps)
        #Third, normalize the input values using the mean and std
        curr_x_cap = (x - curr_mean)/curr_std
        out = gamma * curr_x_cap + beta
        #At each timestep we update the running averages for mean and variance
        #using an exponential decay based on the momentum parameter
        #This will be used during the test phase
        #only to be performed if layer normalization isn't called

        #receive the layernorm info for layer normalization
        layernorm = bn_param.get('layernorm', 0)
        shape = bn_param.get('shape', (N,D)) #reshape will be used for backpassing

        
        if (layernorm == 0): #This will only run if layer normalization flag is off
          running_mean = momentum * running_mean + (1 - momentum) * curr_mean
          running_var = momentum * running_var + (1 - momentum) * curr_variance
        #Last, save all of it in cache for backprop
        cache = (x, curr_x_cap, curr_mean, curr_variance, gamma, eps, layernorm, shape)

        



        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        #We just need to normalize the current values using the running mean and variance
        running_std = np.sqrt(running_var + eps)
        test_x_cap = (x - running_mean)/running_std
        out = gamma * test_x_cap + beta

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #First, we unload the cache tuple
    (x, curr_x_cap, curr_mean, curr_variance, gamma, eps, layernorm, shape) = cache
    
    num_values = 1.0 * x.shape[0] #to be used duirng computations

    

    #Second, we take the gradient at (gamma * test_x_cap) + beta
    #The gradient is split equally
    #summing up the values along the axis 0 for batch norm and 1 for layer norm
    dbeta = dout.reshape(shape, order = 'F').sum(layernorm)
    dgamma = (dout * curr_x_cap).reshape(shape, order = 'F').sum(layernorm)

    #Third, we take the gradient at the normalized input values
    dcurr_x_cap = dout * gamma

    #Following, we take the gradient at curr_x_cap = (x - curr_mean)/curr_std
    #Fourth, gradient at curr_std
    curr_std = np.sqrt(curr_variance + eps)
    dcurr_std = -np.sum(dcurr_x_cap * (x - curr_mean), axis = 0) / (curr_std**2)
    #Fifth, gradient at curr_variance
    dcurr_variance = 0.5 * dcurr_std / curr_std

    #Next, to reach the gradient at x, we need to perform partial differentiation b/w curr_x_cap and curr_mean
    pdx_1 = (dcurr_x_cap / curr_std) + (2 * (x - curr_mean) * dcurr_variance)/ len(dout)
    pdx_2 = (-np.sum(pdx_1, axis = 0)/ len(dout))

    #Add both the partial derivates to arrive at the gradient at x
    dx = pdx_1 + pdx_2

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #First, we unload the cache tuple
    (x, curr_x_cap, curr_mean, curr_variance, gamma, eps, layernorm) = cache
    
    num_values = 1.0 * x.shape[0] #to be used duirng computations

    #Second, we take the gradient at (gamma * test_x_cap) + beta
    #The gradient is split equally
    #summing up the values along the axis 0 for batch norm and 1 for layer norm
    dbeta = np.sum(dout, axis = layernorm)
    dgamma = np.sum(curr_x_cap * dout, axis = layernorm)

    # ##### old implementation, dx difference = 1 ########
    # #Third, we take the gradient at the normalized input values
    # dcurr_x_cap = gamma * dout
    # dx_cap = np.sum(dcurr_x_cap, axis = 0)#add all the values along the axis o

    # #Fourth, we derivate with respect to dx (by hand calculation)
    # dx = (dcurr_x_cap - dx_cap/num_values) - ((np.sum(x * dcurr_x_cap, axis = 0) * x) / num_values)
    # dx = dx/(np.sqrt(curr_variance + eps)) #normalize with the standard deviation
    # #####################

    #### new implementation ###########S
    #Third, we take the gradient at the normalized input values
    dcurr_x_cap = gamma*dout
    #Fourth, we derivatre through the normalization
    dx = dcurr_x_cap / (num_values * (np.sqrt(curr_variance + eps)))
    #Last, we derivate with respect to dx
    dx = num_values*dx - (curr_x_cap *  np.sum((curr_x_cap * dx), axis = 0)) - np.sum(dx, axis = 0)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #First, select the mode as train, since its only in train
    ln_param['mode'] = 'train'
    #Second, set the layer norm flag on
    ln_param['layernorm'] = 1
    #Third, call forward batchnorm, difference being all of the inputs are transposed
    out, cache = batchnorm_forward(x.T, gamma.T, beta.T, ln_param)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out.T, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #First, Call either of the batch norm backward functions
    #We need to transpose the ouput to match the dimensions
    # dx, dgamma, dbeta = batchnorm_backward(dout.T, cache)
    dx, dgamma, dbeta = batchnorm_backward_alt(dout.T, cache) #Call either one
    dx = dx.T #transepose it back to math dimensions

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        #First, to implement an inverted Dropout functionality, we need to create a mask and set 
        #all values below a certain threshold as zero
        #Since it is inverted dropout, we scale the value in train phase only
        mask = (np.random.randn(*x.shape) < p) / p

        #Second, we multiply the incoming value with the mask to eliminate some neurons
        out = x * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        #In test phase, for inverted dropout functionality, we don't do anything at all
        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        #For backprop
        #since drop out acts like a max function (somewhat) to keep certain neurons active
        #we can have the gradients only pass to the existing neurons
        dx = dout * mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #### Reference source for the understanding of forward pass of CNN ###
    # https://towardsdatascience.com/forward-and-backward-propagation-in-convolutional-neural-networks-64365925fdfa 
    ############

    #First, we unpack the stride and padding values form the parameter dictionary
    stride = conv_param['stride']
    pad = conv_param['pad']

    #Second, we unpack the values for the number of data points, channels, height and width of input
    N, _, H, W = x.shape

    #Third, we unpack the filter weights, height and width
    F, _, HH, WW = w.shape

    #In order to work the kernel with processing in the image, padding is added to the outer frame of the input
    #Fourth, we pad the input, zeros placed symmetrically along height and width
    x_pad = np.pad(x, ((0,0), (0,0), (pad, pad), (pad, pad)), 'constant')

    #Fifth, we use the dimensions formula to calculate the width and height of the output
    H_out = int(1 + ((H + 2 * pad - HH)/ stride))
    W_out = int(1 + ((W + 2 * pad - WW)/ stride))

    #Sixth, we initialize an output of zeros
    out = np.zeros((N, F, H_out, W_out))

    #Sixth, we convolve the input with filters to get the output
    for num in range(N):
      for filtr in range(F):
        for heigt in range(H_out):
          for widt in range(W_out):
            out[num, filtr, heigt, widt] = np.sum(x_pad[num, :, heigt*stride:heigt*stride+HH,
            widt*stride:widt*stride+WW] * w[filtr]) + b[filtr]



    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #First, we unpack the cache and then the values from the conv dictionary
    (x, w, b, conv_param) = cache
    stride = conv_param['stride']
    pad = conv_param['pad']

    #Second, we unpack the values for the number of data points, channels, height and width of input
    N, _, H, W = x.shape

    #Third, we unpack the filter weights, height and width
    F, _, HH, WW = w.shape

    #Fourth, we pad the input, zeros placed symmetrically along height and width
    x_pad = np.pad(x, ((0,0), (0,0), (pad, pad), (pad, pad)), 'constant')

    #Fifth, we unpack the values from the dout matrix
    _, _, H_out, W_out = dout.shape

    #Sixth, before we calculate the gradients, we must 0 initialize them in same shapes as the inputs
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    #Now we compute the gradients at each variable
    #W.K.T f = wx+b
    #Since, the gradient splits equally at an addition point, the gradient at d is same as out
    #Seventh, we write the gradient wrt bias, db
    for filtr in range(F):
      db[filtr] = np.sum(dout[:, filtr, :, :]) #summ along the 0 axis to get a value
    
    #Now, the gradient dout, equally falls on wx
    #We know that at a multiplication point, the gradient at one variable is same as the value of the other variable scaled with the upstream gradient
    #Eigth, we frind gradients wrt to x and w
    for num in range(N):
      for filtr in range(F):
        for heigt in range(H_out):
          for widt in range(W_out):
            #gradient wrt weights, dw
            dw[filtr] += x_pad[num, :, heigt*stride:heigt*stride+HH,
            widt*stride:widt*stride+WW] * dout[num, filtr, heigt, widt]
            #gradient wrt padded input, dx_pad
            dx_pad[num, :, heigt*stride:heigt*stride+HH, widt*stride:widt*stride+WW] += w[filtr] * dout[num, filtr, heigt, widt]
    
    #Last, strip the padding from dx_pad to get the gradient at the input, dx
    dx = dx_pad[:, :, pad:-pad, pad:-pad]


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #First, we unpack the pooling parameters
    pool_width = pool_param['pool_width']
    pool_height = pool_param['pool_height']
    stride = pool_param['stride']

    #Second, we unpack the values for the number of data points, channels, height and width of input
    N, C, H, W = x.shape

    #Third, we use the dimensions formula to calculate the width and height of the output
    H_out = int(1 + ((H - pool_height)/ stride))
    W_out = int(1 + ((W - pool_width)/ stride))

    #Fourth, we zero initialize the output
    out = np.zeros([N, C, H_out, W_out])

    #Fifth, we now need to run a filter of poolwidth x poolheight across the input and max pool its values
    for num in range(N):
      for chanl in range(C):
        for heigt in range(H_out):
          for widt in range(W_out):
            out[num, chanl, heigt, widt] = np.max(x[num, chanl, heigt*stride:heigt*stride+pool_height,
            widt*stride:widt*stride+pool_width])


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #First, we unpack the cache and the pooling parameters
    (x, pool_param) = cache
    pool_width = pool_param['pool_width']
    pool_height = pool_param['pool_height']
    stride = pool_param['stride']

    #Second, we unpack the values for the number of data points, channels, height and width of input
    N, C, H, W = x.shape

    #Third, we unpack the values from the dout matrix
    _, _, H_out, W_out = dout.shape

    #Third, to calculate the gradient at x, we must firsst zero initialize it
    dx = np.zeros_like(x)

    #Fourth, we must apply back prop the the max pool layer
    #We know that at a max point, the gradient is routed to the variable containing the max value and the out value gets zero
    for num in range(N):
      for chanl in range(C):
        for heigt in range(H_out):
          for widt in range(W_out):
            #Get the index of the input where the value is maximum
            max_h_index, max_w_index = np.where(np.max(x[num, chanl, heigt*stride:heigt*stride+pool_height,
            widt*stride:widt*stride+pool_width]) == x[num, chanl, heigt*stride:heigt*stride+pool_height, 
            widt*stride:widt*stride+pool_width])
           
            #During backprop only the max indexed values get the gradient, rest stay as zeros
            dx[num, chanl, heigt*stride:heigt*stride+pool_height, widt*stride:widt*stride+pool_width][max_h_index, max_w_index] = dout[num, chanl, heigt, widt]


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #First, we unpack input
    num_values, chanl, heigt, widt = x.shape

    #Second, we adjust the axes to use the vanilla batch-normalization
    new_x = np.swapaxes(x, 0, 1)
    new_x = new_x.reshape(-1, chanl).T #first swap axes and then reshape the matrix
    
    #Third, call the batchnorm forward
    out_batchnorm, cache = batchnorm_forward(new_x, gamma, beta, bn_param)

    #Last, reswap the axes of the output back into original form
    # out = np.moveaxis(out_batchnorm.reshape(num_values, heigt, widt, chanl), -1, 1) #first reshape then moves axes back
    out = out_batchnorm.T.reshape(chanl, num_values, heigt, widt).swapaxes(0,1)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #First, we unpack the gradient at the output
    num_values, chanl, heigt, widt = dout.shape

    #Second, we adjust the axes to use the vanilla batch-normalization
    new_dout = np.moveaxis(dout, 1, -1)
    new_dout = new_dout.reshape(-1, chanl) #first swap axes and then reshape the matrix
    
    #Third, call the batchnorm forward
    dx_backbatchnorm, dgamma, dbeta = batchnorm_backward(new_dout, cache)

    #Last, reswap the axes of the output back into original form
    dx = np.moveaxis(dx_backbatchnorm.reshape(num_values, heigt, widt, chanl), -1, 1) #first reshape then moves axes back
        

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #First, we unpack the input
    num_values, chanl, heigt, widt = x.shape
    #Second, update the shape and axes from the group norm dictionary
    gn_param.update({'shape':(widt, heigt, chanl, num_values), 'axis':(0,1,3)})

    #Third, shape the input values to use vanilla layer normalization
    x = x.reshape(num_values*G, -1)

    #Fourth, reshape the gamma and beta to use vanilla layer normalization
    gamma = np.tile(gamma, (N, 1, H, W)).reshape(N*G, -1)
    beta = np.tile(beta, (N, 1, H, W)).reshape(N*G, -1)

    #Fifth, call the forward layer norm get the output
    out, cache = layernorm_forward(x, gamma, beta, gn_param)
    
    #Last, reshape the output to original dimension
    out = out.reshape(num_values, chanl, heigt, widt)
    cache = (G, cache) #Add G to the cache

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
