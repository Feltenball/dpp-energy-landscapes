# -*- coding: utf-8 -*-
"""
@author: Maria Cann
"""

""" Gram Schmidt Process """

import numpy as np

def projection(v1, v2):
    return np.multiply((np.dot(v1, v2) / np.dot(v1, v1)), v1)

def gram_schmidt(X):
    # Some set of vectors V
    Y = [] # To return
    for i in range(len(X)):
        temp = X[i]
        for prev_vec in Y:
            projected = projection(prev_vec, X[i])
            temp = [temp[i] - projected[i] for i in range(len(temp))]
        Y.append(temp)
    return np.array(Y)