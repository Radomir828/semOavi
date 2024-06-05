import numpy as np
from numpy import log

def haralik(img_arr: np.array, d=2):
    matrix = np.zeros(shape=(256, 256))

    for x in range(d, img_arr.shape[0] - d):
        for y in range(d, img_arr.shape[1] - d):
            matrix[img_arr[x - d, y], img_arr[x, y]] += 1
            matrix[img_arr[x + d, y], img_arr[x, y]] += 1
            matrix[img_arr[x, y - d], img_arr[x, y]] += 1
            matrix[img_arr[x, y + d], img_arr[x, y]] += 1

    for x in range(256):
        m = np.array(matrix[x])
        m[np.where(m == 0)] = 1
        matrix[x] = log(m)
    matrix = matrix * 256 / np.max(matrix)
    return matrix

def CON(har: np.array):
    sum_val = 0
    for i in range(har.shape[0]):
        for j in range(har.shape[1]):
            sum_val += (i - j) ** 2 * har[i, j]
    return sum_val

def LUN(har: np.array):
    sum_val = 0
    for i in range(har.shape[0]):
        for j in range(har.shape[1]):
            sum_val += har[i, j] / (1 + (i - j) ** 2)
    return sum_val
