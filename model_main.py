
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from datetime import datetime

from resnet import Resnet as Model
from DataUtils import *
from BatchTest import BatchTest

# imgfile and labelfile
ROOT = os.getcwd()
print("ROOTpath is {}".format(ROOT))
filepath = os.path.abspath(os.path.join(ROOT, os.pardir))
train_img_path = filepath +  "/imgdata/train/train_x.csv"
train_lab_path = filepath + "/imgdata/train/train_y.csv"
valid_img_path = filepath + "/imgdata/valid/valid_x.csv"
valid_lab_path = filepath + "/imgdata/valid/valid_y.csv"
# filewriter and saver
filewriter_path = filepath + "/tensorboard"
checkpoint_path = filepath + "/checkfile"
datafile_path = filepath + "/result"

if not os.path.isdir(filewriter_path):
    os.mkdir(filewriter_path)
if not os.path.isdir(checkpoint_path):
    os.mkdir(checkpoint_path)

# load image and label files
train_img = loadimg(train_img_path)
train_label = loadlabel(train_lab_path)
valid_img = loadimg(valid_img_path)
valid_label = loadlabel(valid_lab_path)

test_img, test_label = dataShuffle(valid_img, valid_label, shuffle = False)
train_img, train_label = dataShuffle(train_img, train_label, shuffle = True)
valid_img, valid_label = dataShuffle(valid_img, valid_label, shuffle = True)
print("train_img.shape is {}, train_label.shape is {}".format(train_img.shape, train_label.shape))
print("valid_img.shape is {}, valid_label.shape is {}".format(valid_img.shape, valid_label.shape))

# parameters for model
init_learning_rate = 0.001
init_momentum = 0.99
batch_size = 128
num_iterations = 30000
num_classes = 2
num_samples = train_img.shape[0]
size_input = train_img.shape[1]
imgsize = 129
decay_steps = 1000
display_step = 100
csv_steps = 5000

# Iterator for mini-batch
trainimage = tf.placeholder(train_img.dtype, train_img.shape)
trainlabel = tf.placeholder(train_label.dtype, train_label.shape)
train_dataset = tf.data.Dataset.from_tensor_slices((trainimage, trainlabel)).shuffle(buffer_size = 30000).batch(batch_size).repeat(20)
train_iter = train_dataset.make_initializable_iterator()
train_img_batch, train_lab_batch = train_iter.get_next()

validimage = tf.placeholder(valid_img.dtype, valid_img.shape)
validlabel = tf.placeholder(valid_label.dtype, valid_label.shape)
valid_dataset = tf.data.Dataset.from_tensor_slices((validimage, validlabel)).shuffle(buffer_size = 30000).batch(batch_size).repeat(10)
valid_iter = valid_dataset.make_initializable_iterator()
valid_img_batch, valid_lab_batch = valid_iter.get_next()

# model
x = tf.placeholder(tf.float32, [None, size_input])
y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32)
train_mode = tf.placeholder(tf.bool)

model = Model(x, num_classes, keep_prob, train_mode)
out = model.out

with tf.name_scope("prediction"):
    pre = tf.argmax(tf.nn.softmax(out), 1)

with tf.name_scope("Loss"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = out, labels = y))
    tf.summary.scalar("Loss", loss)

# optimizer
with tf.name_scope("Optimizer"):
    global_step = tf.Variable(0, trainable = False)
    learn_rate = tf.train.exponential_decay(init_learning_rate, global_step, decay_steps, 0.96, staircase = True)
    momentum = tf.train.exponential_decay(init_momentum, global_step, 1000, 0.9, staircase = True)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.MomentumOptimizer(learning_rate = learn_rate, momentum = momentum).minimize(loss, global_step = global_step)

with tf.name_scope("Accuracy"):
    correct_pred = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    tf.summary.scalar("Accuracy", acc)


with tf.name_scope("Matrix"):
    y_pred = tf.argmax(out, 1)
    y_actual = tf.argmax(y, 1)
    TP = tf.count_nonzero(y_pred * y_actual)
    TN = tf.count_nonzero((y_pred - 1) * (y_actual - 1))
    FP = tf.count_nonzero(y_pred * (y_actual - 1))
    FN = tf.count_nonzero((y_pred - 1) * y_actual)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    tf.summary.scalar("precision", precision)
    tf.summary.scalar("recall", recall)
    tf.summary.scalar("f1", f1)


init = tf.global_variables_initializer()
merge_summary_op = tf.summary.merge_all()
data_dict = {"Iter": [], "train_loss": [], "train_acc": [], "train_pre": [], "train_recall": [], "train_f1": [],
            "valid_loss": [], "valid_acc": [], "valid_pre": [], "valid_recall": [], "valid_f1": []}
pre_dict = {"label": [], "pre": []}
train_writer = tf.summary.FileWriter(filewriter_path + "/train")
valid_writer = tf.summary.FileWriter(filewriter_path + "/test")
saver_step = 5000
with tf.Session() as sess:
    sess.run(init)
    train_writer.add_graph(sess.graph)
    saver = tf.train.Saver()
    global_step.eval()
    test_iter = int(num_test/batch_size)
    sess.run(train_iter.initializer, feed_dict = {trainimage: train_img, trainlabel: train_label})
    sess.run(valid_iter.initializer, feed_dict = {validimage: valid_img, validlabel: valid_label})

#    ckpt = tf.train.get_checkpoint_state(os.path.dirname(checkpoint_path))
#    if ckpt and ckpt.model_checkpoint_path:
#        saver.restore(sess, ckpt.model_checkpoint_path)
#    else:
#        pass

    for itr in range(num_iterations):
        global_step = itr
        train_imgbatch, train_labatch = sess.run([train_img_batch, train_lab_batch])
        feed_dict = {x: train_imgbatch, y: train_labatch, keep_prob: 0.5, train_mode: True}
        sess.run(optimizer, feed_dict = feed_dict)

        if itr % 10 == 0:
            trs = sess.run(merge_summary_op, feed_dict = feed_dict)
            train_writer.add_summary(trs, itr)
            valid_imgbatch, valid_labatch = sess.run([valid_img_batch, valid_lab_batch])
            vrs = sess.run(merge_summary_op, feed_dict = {x: valid_imgbatch, y: valid_labatch, keep_prob: 1.0, train_mode: False})
            valid_writer.add_summary(vrs, itr)


        if itr % display_step == 0:
            train_imgbatch, train_labatch = sess.run([train_img_batch, train_lab_batch])
            tl, tac, tpre, trec, tfvalue = sess.run([loss, acc, precision, recall, f1], feed_dict = {x: train_imgbatch, y: train_labatch, keep_prob: 0.5, train_mode: True})
            valid_imgbatch, valid_labatch = sess.run([valid_img_batch, valid_lab_batch])
            vl, vac, vpre, vrec, vfvalue, aculabel = sess.run([loss, acc, precision, recall, f1, pre], feed_dict = {x: valid_imgbatch, y: valid_labatch, keep_prob: 1.0, train_mode: False})

            print("-----------Iteration {}-----------".format(itr))
            print("-----------{} starting displaying -----------".format(datetime.now()))
            print("Training Loss is {}, Accuracy is {}".format(tl, tac))
            print("Validing Loss is {}, Accuracy is {}".format(vl, vac))

            data_dict["Iter"].append(itr)
            data_dict["train_loss"].append(tl)
            data_dict["train_acc"].append(tac)
            data_dict["train_pre"].append(tpre)
            data_dict["train_recall"].append(trec)
            data_dict["train_f1"].append(tfvalue)
            data_dict["valid_loss"].append(vl)
            data_dict["valid_acc"].append(vac)
            data_dict["valid_pre"].append(vpre)
            data_dict["valid_recall"].append(vrec)
            data_dict["valid_f1"].append(vfvalue)
            pre_dict["label"].append(sess.run(tf.argmax(valid_labatch, axis = 1)))
            pre_dict["pre"].append(aculabel)

        if itr % saver_step == 0:
            saver.save(sess, checkpoint_path + "/model2_Iter_" + str(itr+1)+ ".ckpt",
                    global_step = global_step)

        if itr % csv_steps == 0:
            data_df = pd.DataFrame(data_dict)
            data_df.to_csv(datafile_path + "/model2_" + str(itr) + ".csv")
            pre_df = pd.DataFrame(pre_dict)
            pre_df.to_csv(datafile_path + "/model2_pre_" + str(itr) + ".csv")
    print("------------------finished---------------------")

