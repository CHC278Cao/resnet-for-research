
# coding: utf-8

# In[ ]:



# this class include the batch Iterator, transform and adding noise
import numpy as np
from hyper_parameters import *


class BatchIt(object):
    def __init__(self, img, label, imgsize, batch_size):
        self.img = img
        self.label = label
        self.imgsize = imgsize
       
        self.index_in_epoch = 0
        self.batch_img, self.batch_label = self.next_batch()

        
    def next_batch(self):
        num_samples = self.img.shape[0]
        start = self.index_in_epoch
        self.index_in_epoch += self.batch_size
        if self.index_in_epoch >= num_samples:
            perm = np.arange(num_samples)
            np.random.shuffle(perm)
            self.img = self.img[perm]
            self.label = self.label[perm]
            start = 0
            self.index_in_epoch = self.batch_size
            assert self.index_in_epoch <= num_sampels
        end = self.index_in_epoch
        img, label = self.img[start: end], self.label[start: end]
        if FLAGS.flip_images:
            img, label = self.transform(img, label)
        if FLAGS.noises_adding:
            img, label = self.add_noise(img, label)
        return img, label

    def transform(self, img, label, rate = 0.5):
        img = img.reshape(-1, self.imgsize, self.imgsize)
        num = img.shape[0]
        indice = np.random.choice(num, int(num * rate), replace = False)
        img[indice] = img[indice, :, ::-1]
        img = img.reshape(-1, self.imgsize * self.imgsize)
        return img, label
    
    def add_noise(self, img, label, rate = 0.5, sigma = 0.1):
        img = img.reshape(-1, self.imgsize, self.imgsize)
        temp_img = np.float64(np.copy(img))
        batch = temp_img.shape[0]
        height = temp_img.shape[1]
        width = temp_img.shape[2]
        
        noise = np.random.randn(batch, height, width) * sigma
        temp_img = temp_img + noise
        temp_img = temp_img.reshape(-1, self.imgsize * self.imgsize)
        
        return temp_img, label
            

