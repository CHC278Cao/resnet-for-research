# -*- coding:utf-8 -*-


# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import argparse

from modelv12 import Resnet
from DataUtils import *

def train(trainFile, validFile, saverPath, checkPath, learningRate, momentum, batchSize,
          iterations, learningDecay, momentumDecay):

    # set the saverPath and checkpointPath to save graph and weight
    if not os.path.isdir(saverPath):
        os.mkdir(saverPath)
    if not os.path.isdir(checkPath):
        os.mkdir(checkPath)

    # load image and label files
    train_img, train_label = concat_data(trainFile)
    valid_img, valid_label = concat_data(validFile)
    print("train_img.shape is {}, train_label.shape is {}".format(train_img.shape, train_label.shape))
    print("valid_img.shape is {}, valid_label.shape is {}".format(valid_img.shape, valid_label.shape))

    # parameters for model
    init_learning_rate = learningRate
    init_momentum = momentum
    batch_size = batchSize
    num_iterations = iterations
    num_classes = 2
    size_input = train_img.shape[1]
    imgsize = 129
    learning_decay = learningDecay
    momentum_decay = momentumDecay
    display_step = 100
    saver_step = 5000

    # Iterator for mini-batch
    trainimage = tf.placeholder(train_img.dtype, train_img.shape)
    trainlabel = tf.placeholder(train_label.dtype, train_label.shape)
    train_dataset = tf.data.Dataset.from_tensor_slices((trainimage, trainlabel)).shuffle(buffer_size = 50000).repeat().batch(batch_size)
    train_iter = train_dataset.make_initializable_iterator()
    train_img_batch, train_lab_batch = train_iter.get_next()

    validimage = tf.placeholder(valid_img.dtype, valid_img.shape)
    validlabel = tf.placeholder(valid_label.dtype, valid_label.shape)
    valid_dataset = tf.data.Dataset.from_tensor_slices((validimage, validlabel)).shuffle(buffer_size = 30000).repeat().batch(batch_size)
    valid_iter = valid_dataset.make_initializable_iterator()
    valid_img_batch, valid_lab_batch = valid_iter.get_next()

    # model placeholder
    x = tf.placeholder(tf.float32, [None, size_input])
    y = tf.placeholder(tf.float32, [None, num_classes])
    keep_prob = tf.placeholder(tf.float32)
    train_mode = tf.placeholder(tf.bool)


    model = Resnet(x, num_classes, imgsize, train_mode, keep_prob)
    out = model.out

    with tf.name_scope("prediction"):
        pre = tf.argmax(tf.nn.softmax(out), 1)

    with tf.name_scope("Loss"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = out, labels = y))
        tf.summary.scalar("Loss", loss)

    # optimizer
    with tf.name_scope("Optimizer"):
        global_step = tf.Variable(0, trainable = False)
        learn_rate = tf.train.exponential_decay(init_learning_rate, global_step, learning_decay, 0.96, staircase = True)
        learn_momentum = tf.train.exponential_decay(init_momentum, global_step, momentum_decay, 0.9, staircase = True)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.MomentumOptimizer(learning_rate = learn_rate, momentum = learn_momentum).minimize(loss, global_step = global_step)

    with tf.name_scope("Accuracy"):
        correct_pred = tf.equal(pre, tf.argmax(y, 1))
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
    train_writer = tf.summary.FileWriter(saverPath + "/train")
    valid_writer = tf.summary.FileWriter(saverPath + "/test")

    with tf.Session() as sess:
        sess.run(init)
        train_writer.add_graph(sess.graph)
        saver = tf.train.Saver()
        global_step.eval()
        sess.run(train_iter.initializer, feed_dict = {trainimage: train_img, trainlabel: train_label})
        sess.run(valid_iter.initializer, feed_dict = {validimage: valid_img, validlabel: valid_label})

        # ckpt = tf.train.get_checkpoint_state(os.path.dirname(checkPath))
        # if ckpt and ckpt.model_checkpoint_path:
        #     saver.restore(sess, ckpt.model_checkpoint_path)
        # else:
        #     pass

        for itr in range(1, num_iterations + 1):
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

            if itr % saver_step == 0:
                saver.save(sess, checkPath + "/model_" + str(itr)+ ".ckpt", global_step = global_step)


        print("------------------finished---------------------")

def main():

    parser = argparse.ArgumentParser(description = 'training the CNN model')
    parser.add_argument('-s', '--saver', dest='saver', help = 'path for saver', default = './tensorboard')
    parser.add_argument('-p', '--ckpoint', dest = 'ckpoint', help = 'path for checkpoint', default = './checkfile')
    parser.add_argument('-r', '--rate', type = float, help = 'initial learning rate', default = 0.001)
    parser.add_argument('-m', '--momentum', type = float, help = 'initial learning momentum', default = 0.99)
    parser.add_argument('-b', '--batch', type = int, help = 'batchsize', default = 64)
    parser.add_argument('-i', '--iteration', type = int, help = 'number of iterations', default = 50000)
    parser.add_argument('-d', '--learningdecay', type = int, help = 'decay steps for learningRate', default = 5000)
    parser.add_argument('-o', '--momentumdecay', type = int, help = 'decay steps for momentumRate', default = 5000)
    # parser.add_argument()

    args = parser.parse_args()
    filepath = "/ihome/tjacobs/chc278/chc278/imgdata"
    trainfile = [filepath + "/train2/train" + str(i) + ".csv"  for i in range(1, 33)]
    validfile = [filepath + "/valid/valid_" + str(i) + ".csv" for i in range(1, 4)]
    train(trainfile, validfile, args.saver, args.ckpoint, args.rate, args.momentum, args.batch,
          args.iteration, args.learningdecay, args.momentumdecay)

if __name__ == '__main__':
    main()
