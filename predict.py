
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from datetime import datetime

from modelv2 import Model
from DataUtils import *
from pixelSeg import pixelSeg
import cv2

# imgfile and labelfile
ROOT = os.getcwd()
print("ROOTpath is {}".format(ROOT))
filepath = os.path.abspath(os.path.join(ROOT, os.pardir))
test_img_path = filepath + "/imgdata/image/img3.tif"

# filewriter and saver
filewriter_path = filepath + "/tensorboard"
checkpoint_path = filepath + "/checkfile"
datafile_path = filepath + "/result"

# parameters for testdata subimage
padsize = 64
imgsize = 129
mode = cv2.BORDER_REPLICATE
start = 2100
end = 2112

# load image and label files
test = pixelSeg(test_img_path, padsize, mode, imgsize, start, end)
test_img = test.pixelimg

print("test_img.shape is {}".format(test_img.shape))

# parameters for test model
num_test = test_img.shape[0]
test_batchsize = 2048
num_test = int(num_test/test_batchsize)

# predict label
pre_dict = []

# load the checkpoint and graph
tf.reset_default_graph()
new_graph = tf.Graph()
with tf.Session(graph = new_graph) as sess:
    model_saver = tf.train.import_meta_graph(checkpoint_path + "/model2_Iter_20001.ckpt-20000.meta")
    model_saver.restore(sess, checkpoint_path + "/model2_Iter_20001.ckpt-20000")

    # get the default graph and then use the "get_tensor_by_name" method
    graph = tf.get_default_graph()

    x = graph.get_tensor_by_name("Placeholder_4:0")
    y = graph.get_tensor_by_name("Placeholder_5:0")
    keep_prob = graph.get_tensor_by_name("Placeholder_6:0")
    train_mode = graph.get_tensor_by_name("Placeholder_7:0")

    output = graph.get_tensor_by_name("prediction/ArgMax:0")

    # visvualize the weights in some layers
#    weights = graph.get_tensor_by_name("Conv2d_0a_3x3/weights:0")
#    print(sess.run(weights))

    # print the all op_name in the defined graph
#    for op in graph.get_operations():
#        print(op.name)

    # print the collection of tranable_variables
#    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
#    for v in train_vars:
#        if v.name.endswith("weights:0"):
#            print(v)

    # print the summaries in the defined graph
    summaries_vars = tf.get_collection(tf.GraphKeys.SUMMARIES)
    for var in summaries_vars:
        print(var)
    print("===================== restore model ========================")

    for i in range(num_test):
        test_imgbatch = test_img[i*test_batchsize: (i+1)*test_batchsize]
        if i == 0:
            print(test_imgbatch.shape)
        pred_label = sess.run(output, feed_dict = {x: test_imgbatch, keep_prob: 1.0, train_mode: False})
        pre_dict.append(pred_label)
        if i == 5:
            print(pred_label)

    pre_dict = np.squeeze(pre_dict)
    pre_df = pd.DataFrame(pre_dict)
    pre_df.to_csv(datafile_path + "/img3_test_" + str(end) + ".csv", index = False)

    print("------------------finished---------------------")

