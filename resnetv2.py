
# coding: utf-8

# In[ ]:


from netUtils import *
import tensorflow as tf
import numpy as np

class Resnet(object):
    def __init__(self, inputs, num_classes, imgsize, is_training, keep_prob):
        self.inputs = tf.reshape(inputs, [-1, imgsize, imgsize, 1])
        self.num_classes = num_classes
        self.is_training = is_training
        self.keep_prob = keep_prob

        self._buildModel(self.is_training, self.keep_prob)

    def _buildModel(self, train_mode, keep_prob):
        # inputs is 129x129x1
        with tf.variable_scope("conv0"):
            conv0 = conv_bn_relu_layer(self.inputs, 8, 3, 1, padding = "VALID", is_training = train_mode)
            activation_summary(conv0)

        # feature map is 127x127x8
        with tf.variable_scope("pool0"):
            pool0 = pool2d(conv0, ksize = 2, stride = 2)
            activation_summary(pool0)

        # feature map is 64x64x8
        with tf.variable_scope("conv1"):
            conv1 = block_v1(pool0, 8, 8, 16, train_mode)
            activation_summary(conv1)

        # feature map is 64x64x16
        with tf.variable_scope("conv2"):
            conv2 = resdual_v1(conv1, 16, 16, 16, train_mode)
            activation_summary(conv2)

        # feature map is 64x64x32
        with tf.variable_scope("conv3"):
            conv3 = resdual_v2(conv2, 16, 16, 32, train_mode)
            activation_summary(conv3)

        # feature map is 32x32x32
        with tf.variable_scope("conv4"):
            conv4 = resdual_v1(conv3, 16, 16, 32, train_mode)
            activation_summary(conv4)

        # feature map is 32x32x64
        with tf.variable_scope("conv5"):
            conv5 = resdual_v2(conv4, 16, 16, 64, train_mode)
            activation_summary(conv5)

        # feature map is 16x16x64
        with tf.variable_scope("conv6"):
            conv6 = resdual_v1(conv5, 32, 32, 64, train_mode)
            activation_summary(conv6)

        # feature map is 16x16x64
        with tf.variable_scope("conv7"):
            conv7 = resdual_v2(conv6, 32, 32, 64, train_mode)
            activation_summary(conv7)

        # feature map is 16x16x64
        with tf.variable_scope("conv8"):
            conv8 = resdual_v1(conv7, 32, 32, 64, train_mode)
            activation_summary(conv8)

        # feature map is 16x16x64
        with tf.variable_scope("conv9"):
            conv9 = resdual_v2(conv8, 32, 32, 64, train_mode)
            activation_summary(conv9)

        # feature map is 8x8x64
        with tf.variable_scope("conv10"):
            w = create_conv_variables(name = "weights", shape = [1, 1, 64, 64])
            b = create_bias_variables(name = "biases", shape = [64])
            conv10 = conv2d(conv9, w, b, stride = 1 , padding = "SAME")
            activation_summary(conv10)

        # feature map is 8x8x64
        shape = conv10.get_shape().as_list()
        flatten  = tf.reshape(conv10, shape = [-1, shape[1]*shape[2]*shape[3]])

        with tf.variable_scope("fc1"):
            fc1 = fc_layer(flatten, 500)
            fc1 = tf.cond(train_mode, lambda: tf.nn.dropout(fc1, keep_prob), lambda: fc1)
            activation_summary(fc1)

        with tf.variable_scope("fc2"):
            fc2 = fc_layer(fc1, 500)
            fc2 = tf.cond(train_mode, lambda: tf.nn.dropout(fc2, keep_prob), lambda: fc2)
            activation_summary(fc2)

        with tf.variable_scope("fc3"):
            fc3 = fc_layer(fc2, 200)
            fc3 = tf.cond(train_mode, lambda: tf.nn.dropout(fc3, keep_prob), lambda: fc3)
            activation_summary(fc3)

        with tf.variable_scope("fc4"):
            fc4 = fc_layer(fc3, 200)
            fc4 = tf.cond(train_mode, lambda: tf.nn.dropout(fc4, keep_prob), lambda: fc4)
            activation_summary(fc4)

        with tf.variable_scope("fc5"):
            fc5 = fc_layer(fc4, self.num_classes)
            fc5 = tf.cond(train_mode, lambda: tf.nn.dropout(fc5, keep_prob), lambda: fc5)
            activation_summary(fc5)

        self.out = fc5

