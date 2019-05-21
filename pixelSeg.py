# create the segment pixel subimg
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import cv2

class pixelSeg(object):
    def __init__(self, imgpath, padsize, mode, imgwidth, start, end):
        """
        Args:
            imgpath: original imagepath
            padsize: the desired padding size to be added to the images
            mode: the mode to add padding, like REPLICATE, REFLECT
            imgwidth: the width of subimage
            start: start point for subimage, from padsize to imgshape+padsize
            end: end point for subiamge, from padsize to imgshape+padsize
        """
        self.imgpath = imgpath
        self.padsize = padsize
        self.mode = mode
        self.imgwidth = imgwidth
        self.start = start
        self.end = end

        self.img = cv2.imread(self.imgpath, 0)
        self.rep_img = self._addPadding(self.img, self.padsize, self.mode)

        self.pixelimg = self._pixelImg(self.rep_img, self.padsize, self.imgwidth, self.start, self.end)

    def _addPadding(self, img, padsize, mode):
        """
            Adding the padding to the original images, and return the new image
        """
        rep_img = cv2.copyMakeBorder(img, padsize, padsize, padsize, padsize, mode)
        print("rep_img.shape is {}".format(rep_img.shape))
        return rep_img

    def _pixelImg(self, img, padsize, imgwidth, start, end):
        """
            Extract the subimage from the new image which has the added padding,
            and cause the img is big, then each time run the code, change the range
            of i, and it will crop img from left to right
        """
        hlimit = img.shape[0] - padsize
        wlimit = img.shape[1] - padsize
        tol = img.shape[0]
        width = math.floor(imgwidth/2)
        imglist = []
        # change the range of i, keep the range of j
        for i in range(start, end):
            for j in range(padsize, wlimit):
                assert (i-width >=0 and i+width+1 <= tol and j-width >= 0 and j+width+1 <= tol)
                segimg = img[i-width: i+width+1, j-width: j+width+1]
                segimg = segimg.reshape(-1, self.imgwidth * self.imgwidth)
                segimg = np.multiply(segimg, 1.0/225.0)

                imglist.append(segimg)

        img = np.squeeze(imglist)
        return img
