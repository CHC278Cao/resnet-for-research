# -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import scipy.io
import cv2


def interpolation(x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0
    data = []

    if (x0 == x1):
        yk = y0 if y0 < y1 else y1
        for i in range(dy + 1):
            data.append((x0, yk))
            yk += 1
        return data

    m = float(dy) / dx
    dx, dy = abs(dx), abs(dy)
    tdx, tdy = 2 * dx, 2 * dy

    if (m > 1):
        p = 2 * dx - dy
        xk = x0 if x0 < x1 else x1
        yk = y0 if y0 < y1 else y1
        for i in range(dy + 1):
            data.append((xk, yk))
            yk += 1
            if (p >= 0):
                xk += 1
                p -= tdy
            p += tdx


    elif (m == 1):
        xk = x0 if x0 < x1 else x1
        yk = y0 if y0 < y1 else y1
        for i in range(dx + 1):
            data.append((xk, yk))
            xk += 1
            yk += 1

    elif (m > 0 and m < 1):
        p = 2 * dy - dx
        xk = x0 if x0 < x1 else x1
        yk = y0 if y0 < y1 else y1
        for i in range(dx + 1):
            data.append((xk, yk))
            xk += 1
            if (p >= 0):
                yk += 1
                p -= tdx
            p += tdy

    elif (m > -1 and m < 0):
        p = 2 * dy - dx
        xk = x0 if x0 > x1 else x1
        yk = y0 if y0 < y1 else y1
        for i in range(dx + 1):
            data.append((xk, yk))
            xk -= 1
            if (p >= 0):
                yk += 1
                p -= tdx
            p += tdy

    elif (m == -1):
        xk = x0 if x0 < x1 else x1
        yk = y0 if y0 > y1 else y1
        for i in range(dx + 1):
            data.append((xk, yk))
            xk += 1
            yk -= 1

    elif (m < -1):
        p = 2 * dx - dy
        xk = x0 if x0 < x1 else x1
        yk = y0 if y0 > y1 else y1
        for i in range(dy + 1):
            data.append((xk, yk))
            yk -= 1
            if (p >= 0):
                xk += 1
                p -= tdy
            p += tdx

    elif (m == 0):
        xk = x0 if x0 < x1 else x1
        for i in range(dx + 1):
            data.append((xk, y0))
            xk += 1

    return data


def gtDataOps(dataPath):
    df = scipy.io.loadmat(dataPath)
    data = df['pos'].tolist()
    print(len(data))


    interData = []
    for i in range(len(data) - 1):
        if data[i][0] <= 0 or data[i][0] > 2048 or data[i][1] <= 0 or data[i][1] > 2048:
            continue
        begin_x = data[i][0]
        begin_y = data[i][1]
        end_x = data[i + 1][0]
        end_y = data[i + 1][1]
        print("begin point is ({0}, {1}), end point is ({2}, {3})".format(begin_x, begin_y, end_x, end_y))
        temp = interpolation(begin_x, begin_y, end_x, end_y)
        for x in temp:
            if x[0] <= 0 or x[0] > 2048 or x[1] <= 0 or x[1] > 2048 or (x in interData):
                continue
            interData.append(x)
    print("The number of border point is {}".format(len(interData)))

    return interData

def gtImgOps(imgpath, pointData):

    original_img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    (h, w) = np.shape(original_img)
    gtimg = np.zeros((h, w), dtype = 'uint8')
    for i in range(len(pointData)):
        gtimg[pointData[i][1], pointData[i][0]] = 1
    # print("the number of border point: {}".format(sum(sum(gtimg))))
    fig, axes = plt.subplots(1, 2, figsize = (10, 10))
    axes[0].imshow(original_img, cmap = 'gray')
    axes[1].imshow(gtimg, cmap = 'gray')
    plt.show()

    return gtimg


def txtOps(txtPath):
    digData = []

    with open(txtPath, 'r') as f:
        content = f.readlines()
    for line in content:
        line = line.strip()
        temp = re.findall(r"[-+]?\d+\.?\d+", line)
        digData.append(temp[0])

    scalar = float(digData[-2])
    degree = float(digData[-1])
    return (scalar, degree)


def dataOps(imgPath, dataPath, txtPath, gtDataPath, gtImgPath):
    gtdata = gtDataOps(dataPath)
    (scalar, degree) = txtOps(txtPath)
    gtimg = gtImgOps(imgPath, gtdata)

    dataDf = pd.DataFrame(gtdata)
    dataDf.to_csv(gtDataPath, index = False)
    imgDf = pd.DataFrame(gtimg)
    imgDf.to_csv(gtImgPath, index = False)


def main():
    """
    imgPath: original images path
    dataPath: data position after being modified, file type should be ".csv"
    txtPath: txtfile path includes scalar and degree
    gtDataPath: saving the new border position after interpolating
    gtImgPath: the ground-truth image of original image
    :return:
    """
    imgPath = sys.argv[1]
    dataPath = sys.argv[2]
    txtPath = sys.argv[3]
    gtDataPath = sys.argv[4]
    gtImgPath = sys.argv[5]

    for file in (imgPath, dataPath, txtPath):
        if os.stat(file).st_size == 0:
            raise ValueError("{} doesn't exit ...".format(file))

    dataOps(imgPath, dataPath, txtPath, gtDataPath, gtImgPath)


if __name__ == "__main__":
    main()