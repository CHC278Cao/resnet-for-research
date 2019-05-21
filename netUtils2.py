import numpy as np
import tensorflow as tf


WEIGHT_DECAY = 1e-5


def create_conv_variables(name, shape, weight_decay = True,
                          init =tf.contrib.layers.xavier_initializer()):
    """
        generate the convolutional layer weights
    Args:
        name: variables name
        shape: weight tensor shape
        init: weight initializd method, using xavier initializer by default
    """
    if weight_decay:
        regularizer = tf.contrib.layers.l2_regularizer(scale = WEIGHT_DECAY)
        var = tf.get_variable(name, shape = shape, initializer = init, regularizer = regularizer)
    else:
        var = tf.get_variable(name, shape = shape, initializer = init)

    return var

def create_fc_variables(name, shape, weight_decay = True,
                        init = tf.truncated_normal_initializer(0.0, stddev = 0.01)):
    """
        generate the fc Layer weights
    Args:
        name: variables name
        shape: weight tensor shape
        init: weight initializd method, using truncated normal initializer by default
    """
    if weight_decay:
        regularizer = tf.contrib.layers.l2_regularizer(WEIGHT_DECAY)
        var = tf.get_variable(name, shape = shape, initializer = init, regularizer = regularizer)
    else:
        var = tf.get_variable(name, shape = shape, initializer = init)

    return var

def create_bias_variables(name, shape, init = tf.constant_initializer(0.0)):
    """
        generate the bias variables
    Args:
        name: variables name
        shape: bias tensor shape
        init: bias initializd method, using constant initializer by default
    """
    var = tf.get_variable(name, shape = shape, initializer = init)
    return var

def conv2d(inputs, w, stride, b = None, padding = "VALID"):
    """
        calculate the convolutional output
    Args:
        inputs: 4-D tensor
        w: weight for convolution layer
        b: bias for convolution layer
        stride: stride for convolution
        padding: "VALID" by default, or "SAME"
    """
    if b is not None:
        conv = tf.nn.conv2d(inputs, w, strides = [1, stride, stride, 1], padding = padding)
        out = tf.nn.bias_add(conv, b)
    else:
        out = tf.nn.conv2d(inputs, w, strides = [1, stride, stride, 1], padding = padding)
    return out

def conv2_atrous(inputs, w, rate, b, padding):
    """
        calculate the atrous convolutional output
    Args:
        inputs: 4-D tensor
        w: weight for atrous convolution layer, the new filter shape is [filter_height+(filter_height-1)*(rate-1), filter_width+(filter_width-1)*(rate-1)]
        rate: dilate rate
        padding: "VALID" for default
    """
    if b is not None:
        conv = tf.nn.atrous_conv2d(inputs, w, rate = rate, padding = padding)
        out = tf.nn.bias_add(conv, b)
    else:
        out = tf.nn.atrous_conv2d(inputs, w, rate = rate,padding = padding)

    return out

def pool2d(inputs, ksize, stride, padding = "SAME", active_func = tf.nn.max_pool):
    """
        calculate the pooling layer output
    Args:
        inputs: 4-D tensor
        ksize: window size for pooling
        stride: stride for pooling
        active_func: pooling type, using tf.nn.max_pool by default, or tf.nn.avg_pool
    """
    return active_func(inputs, ksize = [1, ksize, ksize, 1], strides = [1, stride, stride, 1], padding = padding)

def fc_layer(inputs, out_dim):
    """
        calculate the out of the model
        inputs: the fc Layer, 2-D tensor
        num_classes: number of the classes
    """
    input_dim = inputs.get_shape().as_list()[-1]
    fc_w = create_fc_variables(name = "fc_w", shape = [input_dim, out_dim])
    fc_b = create_bias_variables(name = "fc_b", shape = [out_dim])
    out = tf.matmul(inputs, fc_w) + fc_b
    return out


def batch_normalization_defined(inputs, dimension, is_training = True, decay = 0.9, eps = 1e-5, name = "bn"):
    """
        batch normalization for layers
    Args:
        inputs: inputs for batch normalization
        dimension: layer.get_shape().as_list()[-1]
    """
    with tf.variable_scope(name):
        beta = tf.get_variable("beta", dimension,
                               initializer = tf.constant_initializer(0.0, dtype = tf.float32), trainable = False)
        gamma = tf.get_variable("gamma", dimension,
                                initializer = tf.constant_initializer(1.0, dtype = tf.float32), trainable = False)
        batch_mean, batch_var = tf.nn.moments(inputs, axes = [0, 1, 2], name = "moments")
        ema = tf.train.ExponentialMovingAverage(decay = decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(is_training, mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        bn_layer = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, eps = eps)
    return bn_layer


def batch_normalization(inputs, is_training, decay = 0.997, eps = 1e-5):
    """
        batch normalization for inputs
    Args:
        inputs: 4-D tensor
        is_training: if training, the mean and avariance will update, otherwise, it won't update
    """
    return tf.contrib.layers.batch_norm(inputs, decay = decay, epsilon = eps, is_training = is_training)

def conv_bn_relu_layer(inputs, out_dim, kernel, stride, padding, is_training):
    """
        generate the result when the inputs is feeded to conv, bn, relu sequentially
    Args:
        inputs: input 4-D tensor
        kernelsize: kernel size for convolutional layer
        stride: stride for co convolution
        padding: padding for convolution layer
        is_training: if training, the mean and avariance will update, otherwise, it won't update
    """
    in_dim = inputs.get_shape().as_list()[-1]
    w = create_conv_variables(name = "weights", shape = [kernel, kernel, in_dim, out_dim])
    b = create_bias_variables(name = "baises", shape = [out_dim])
    conv = conv2d(inputs, w, stride = stride, padding = padding)
    bn = batch_normalization(conv, is_training = is_training)
    out = tf.nn.bias_add(bn, b)
    output = tf.nn.relu(out)

    return output

def atrous_conv_bn_relu_layer(inputs, out_dim, kernel, rate, padding = "VALID", is_training = "None"):
    in_dim = inputs.get_shape().as_list()[-1]
    w = create_conv_variables(name = "weights", shape = [kernel, kernel, in_dim, out_dim])
    b = create_bias_variables(name = "biases", shape = [out_dim])
    conv = conv2_atrous(inputs, w, rate, b = None, padding = padding)
    bn = batch_normalization(conv, is_training = is_training)
    out = tf.nn.bias_add(bn, b)
    output = tf.nn.relu(out)

    return output


def bn_relu_conv_layer(inputs, out_dim, kernel, stride, is_training):
    in_dim = inputs.get_shape().as_list()[-1]
    bn = batch_normalization(inputs, is_training = is_training)
    relu = tf.nn.relu(bn)
    w = create_conv_variables(name = "weights", shape = [kernel, kernel, in_dim, out_dim])
    b = create_bias_variables(name = "baises", shape = [out_dim])
    conv = conv2d(inputs, w, stride = stride, b = b, padding = "SAME")

    return conv

def block_v1(inputs, out_dim_bn1, out_dim_bn2, out_dim_bn3, is_training):
    """
        residual block for resnet, this block won't decrease the size of feature map
    Args:
        inputs: input 4-D tensor
        out_dim_bn1: the first conv_bn_relu layer feature map size
        out_dim_bn2: the second conv_bn_relu layer feature map size
        out_dim_bn3: the third conv_bn_relu layer feature map size
        is_training: the setting option for batch normalization
    """
    in_dim = inputs.get_shape().as_list()[-1]
    with tf.variable_scope("Branch_1"):
        out1 = conv_bn_relu_layer(inputs, out_dim_bn1, kernel = 1, stride = 1, padding = "SAME", is_training = is_training)
    with tf.variable_scope("Branch_2"):
        out2 = conv_bn_relu_layer(out1, out_dim_bn2, kernel = 3, stride = 1, padding = "SAME", is_training = is_training)
    with tf.variable_scope("Branch_3"):
        w = create_conv_variables(name = "weights", shape = [1, 1, out_dim_bn2, out_dim_bn3])
        b = create_bias_variables(name = "baises", shape = [out_dim_bn3])
        conv3 = conv2d(out2, w, stride = 1, padding = "SAME")
        bn3 = batch_normalization(conv3, is_training = is_training)
        bn3 = tf.nn.bias_add(bn3, b)
    with tf.variable_scope("Branch_4"):
        w = create_conv_variables(name = "weights", shape = [1, 1, in_dim, out_dim_bn3])
        b = create_bias_variables(name = "baises", shape = [out_dim_bn3])
        conv4 = conv2d(inputs, w, stride = 1, padding = "SAME")
        bn4 = batch_normalization(conv4, is_training = is_training)
        bn4 = tf.nn.bias_add(bn4, b)

    out = bn4 + bn3
    out = tf.nn.relu(out)

    return out


def resdual_v1(inputs, out_dim_bn1, out_dim_bn2, out_dim_bn3, is_training):
    """
        residual block for resnet, this block won't decrease the size of feature map
    Args:
        inputs: input 4-D tensor
        out_dim_bn1: the first conv_bn_relu layer feature map size
        out_dim_bn2: the second conv_bn_relu layer feature map size
        out_dim_bn3: the third conv_bn_relu layer feature map size
        is_training: the setting option for batch normalization
    """
    in_dim = inputs.get_shape().as_list()[-1]
    with tf.variable_scope("Branch_1"):
        out1 = conv_bn_relu_layer(inputs, out_dim_bn1, kernel = 1, stride = 1, padding = "SAME", is_training = is_training)
    with tf.variable_scope("Branch_2"):
        out2 = conv_bn_relu_layer(out1, out_dim_bn2, kernel = 3, stride = 1, padding = "SAME", is_training = is_training)
    with tf.variable_scope("Branch_3"):
        w = create_conv_variables(name = "weights", shape = [1, 1, out_dim_bn2, out_dim_bn3])
        b = create_bias_variables(name = "baises", shape = [out_dim_bn3])
        conv3 = conv2d(out2, w, stride = 1, padding = "SAME")
        bn3 = batch_normalization(conv3, is_training = is_training)
        bn3 = tf.nn.bias_add(bn3, b)

    out = inputs + bn3
    out = tf.nn.relu(out)

    return out

def resdual_v2(inputs, out_dim_bn1, out_dim_bn2, out_dim_bn3, is_training):
    """
        residual block, this will turn the feature map to a half
    Args:
        inputs: input 4-D tensor
        out_dim_bn1: the first conv_bn_relu layer feature map size
        out_dim_bn2: the second conv_bn_relu layer feature map size
        out_dim_bn3: the third conv_bn_relu layer feature map size
        is_training: the setting option for batch normalization
    """
    in_dim = inputs.get_shape().as_list()[-1]
    with tf.variable_scope("Branch_1"):
        out1 = conv_bn_relu_layer(inputs, out_dim_bn1, kernel = 1, stride = 2, padding = "SAME",  is_training = is_training)
    with tf.variable_scope("Branch_2"):
        out2 = conv_bn_relu_layer(out1, out_dim_bn2, kernel = 3, stride = 1, padding = "SAME", is_training = is_training)
    with tf.variable_scope("Branch_3"):
        w = create_conv_variables(name = "weights", shape = [1, 1, out_dim_bn2, out_dim_bn3])
        b = create_bias_variables(name = "baises", shape = [out_dim_bn3])
        conv3 = conv2d(out2, w, stride = 1, padding = "SAME")
        bn3 = batch_normalization(conv3, is_training = is_training)
        bn3 = tf.nn.bias_add(bn3, b)
    with tf.variable_scope("Branch_4"):
        w = create_conv_variables(name = "weights", shape = [1, 1, in_dim, out_dim_bn3])
        b = create_bias_variables(name = "baises", shape = [out_dim_bn3])
        conv4 = conv2d(inputs, w, stride = 2, padding = "SAME")
        bn4 = batch_normalization(conv4, is_training = is_training)
        bn4 = tf.nn.bias_add(bn4, b)

    out = bn3 + bn4
    out = tf.nn.relu(out)

    return out

def activation_summary(x):
    """
        histogram summary and scalar summary of the ops, activations to see the activation of weights,
        sparsity to see the rate of zero in output
    """
    op_name = x.op.name
    tf.summary.histogram(op_name + "/activations", x)
    tf.summary.scalar(op_name + "/sparsity", tf.nn.zero_fraction(x))

