
# coding: utf-8

# In[ ]:


from netUtils2 import *
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
            pool0 = conv_bn_relu_layer(conv0, 8, 3, 1, padding = "SAME", is_training = train_mode)
            activation_summary(pool0)

        # feature map is 64x64x8
        with tf.variable_scope("conv1"):
            conv1 = block_v1(pool0, 8, 8, 16, train_mode)
            activation_summary(conv1)

        # feature map is 64x64x16
        with tf.variable_scope("conv2"):
            conv2 = resdual_v1(conv1, 16, 16, 16, train_mode)
            activation_summary(conv2)

        # feature map is 64x64x16
        with tf.variable_scope("conv3"):
            conv3 = resdual_v2(conv2, 16, 16, 32, train_mode)
            activation_summary(conv3)

        # feature map is 32x32x32
        with tf.variable_scope("conv4"):
            conv4 = resdual_v1(conv3, 16, 16, 32, train_mode)
            activation_summary(conv4)

        # feature map is 32x32x32
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

        # feature map is 8x8x64
        with tf.variable_scope("conv8"):
            conv8 = resdual_v1(conv7, 32, 32, 64, train_mode)
            activation_summary(conv8)

        # feature map is 16x16x64
        with tf.variable_scope("add_conv1"):
            add_conv1 = atrous_conv_bn_relu_layer(conv8, 64, 3, 2, is_training = train_mode)
            activation_summary(add_conv1)

        # feature map is 12x12x64
        with tf.variable_scope("add_conv2"):
            add_conv2 = atrous_conv_bn_relu_layer(add_conv1, 64, 3, 2, is_training = train_mode)
            activation_summary(add_conv2)

        # feature map is 8x8x64
        with tf.variable_scope("conv9"):
            w = create_conv_variables(name = "weights", shape = [1, 1, 64, 64])
            b = create_bias_variables(name = "biases", shape = [64])
            conv9 = conv2d(add_conv2, w, stride = 1, b = b, padding = "SAME")
            conv9 = tf.nn.relu(conv9)
            activation_summary(conv9)

        # feature map is 8x8x64
        with tf.variable_scope("con10"):
            conv10 = pool2d(conv9, ksize = 8, stride = 1, padding = "VALID", active_func = tf.nn.avg_pool)
            conv10 = tf.nn.relu(conv10)
            activation_summary(conv10)

        # feature map is 1x1x64
        with tf.variable_scope("conv11"):
            w = create_conv_variables(name = "weights", shape = [1, 1, 64, 64])
            b = create_bias_variables(name = "biases", shape = [64])
            conv11 = conv2d(conv10, w, stride = 1, b = b, padding = "VALID")
            conv11 = tf.nn.relu(conv11)
            activation_summary(conv11)

        flatten = tf.squeeze(conv11, [1, 2])

        with tf.variable_scope("fc"):
            out = fc_layer(flatten, self.num_classes)
            activation_summary(out)

        self.out = out

