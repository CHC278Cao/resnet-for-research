# -*- coding:utf-8 -*-

import os
import numpy as np
import cv2
import pandas as pd

def postcess(onesfile, zerosfile, flag, shreshold = None, kernelsize = None):
    output = []
    if flag == 0:
        res = np.subtract(onesfile,  zerosfile)
        output = [[1 if x > 0 else 0 for x in row] for row in res]
    elif flag == 1:
        assert (shreshold is not None)
        output = [[1 if x >= shreshold else 0 for x in row] for row in onesfile]
    elif flag == 2:
        assert (kernelsize is not None)
        res = np.subtract(onesfile - zerosfile)
        temp = [[1 if x > 0 else 0 for x in row] for row in res]
        kernel = np.ones((kernelsize, kernelsize), np.uint8)
        erosion = cv2.erode(temp, kernel, iterations = 1)
        output = erosion
    else:
        raise ValueError("Using wrong option for flag, it should be (0, 1, 2)")

    output = np.array(output, dtype = 'uint8')

    return output


def figPlot(images, saverPath):

    path = saverPath.split('/')
    path[-1] = path[-1].split('.')[0] + '.png'
    saver = '/'.join(path)
    fig = plt.figure(figsize = (12, 12))
    plt.imshow(images, cmap = 'gray')
    fig.savefig(saver, dpi = fig.dpi)
    plt.show()





