
# coding: utf-8

# In[ ]:


from resnetUtils import *

class Resnet(object):
    def __init__(self, inputs, num_classes, imgsize, is_training):
        self.inputs = tf.reshape(inputs, [-1, imgsize, imgsize, 1])
        self.num_classes = num_classes
        self.is_training = is_training
        
    def _buildModel(self):
        # inputs is 129x129x1
        with tf.variable_scope("conv0"):
            conv0 = conv_bn_relu_layer(self.inputs, 8, 3, 1, padding = "VALID", train_mode = self.is_training)
            activation_summary(conv0)
            
        # feature map is 127x127x8
        with tf.variable_scope("pool0"):
            pool0 = pool2d(conv0, ksize = 2, stride = 2)
            activation_summary(pool0)
        
        # feature map is 64x64x8
        with tf.variable_scope("conv1"):
            conv1 = block_v1(pool0, 8, 8, 16, self.is_training)
            activation_summary(conv1)
        
        # feature map is 64x64x16
        with tf.variable_scope("conv2"):
            conv2 = resdual_v1(conv1, 16, 16, 32, self.is_training)
            activation_summary(conv2)
            
        # feature map is 64x64x32
        with tf.variable_scope("conv3"):
            conv3 = resdual_v2(conv2, 16, 16, 32, self.is_training)
            activation_summary(conv3)
        
        # feature map is 32x32x32
        with tf.variable_scope("conv4"):
            conv4 = resdual_v1(conv3, 32, 32, 64, self.is_training)
            activation_summary(conv4)
            
        # feature map is 32x32x64
        with tf.variable_scope("conv5"):
            conv5 = resdual_v2(conv4, 32, 32, 64, self.is_training)
            activation_summary(conv5)
        
        # feature map is 16x16x64
        with tf.variable_scope("conv6"):
            conv6 = resdual_v1(conv5, 32, 32, 64, self.is_training)
            activation_summary(conv6)
        
        # feature map is 16x16x64
        with tf.variable_scope("conv7"):
            conv7 = resdual_v2(conv6, 64, 64, 64, self.is_training)
            activation_summary(conv7)
        
        # feature map is 8x8x64
        with tf.variable_scope("pool8"):
            pool8 = pool2d(conv7, ksize = 8, stride = 1, padding = "VALID", active_func = tf.nn.avg_pool)
        
        # feature map is 1x1x64
        with tf.variable_scope("fc"):
            in_dim = pool8.get_shape().as_list()[-1]
            global_pool = tf.reduce_mean(pool8, [1, 2])
            output = output_layer(global_pool, self.num_classes)
        
        self.out = output

