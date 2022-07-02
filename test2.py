# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 08:41:52 2022

@author: Maria
"""

import numpy as np
import matplotlib.pyplot as plt
import sample_LDPP

""" L_ij = q_i S_ij q_j 
S(x, y) = exp(-1/2s^2 |x-y|^2)
N = # grid points 
dim = space dimension 
Y = ground set 
L = L kernel 

Here we set the quality coefficients to all be 1. 
Only care about diversity model for this test.
"""

N = 25
dim = 2
sigma = 15
Y = np.zeros(shape=(N**2, dim))
index = 0
for i in range(0, N):
    for j in range(0, N):
        Y[index, :] = [i, j]
        index += 1
        
L = np.zeros(shape=(N**2, N**2))
for i in range(0, N**2):
    for j in range(0, N**2):
        L[i][j] = np.exp(-1/(2 * sigma) * np.linalg.norm(Y[i][:] 
                                                         - Y[j][:])**2)

evals, evecs = np.linalg.eig(L)
evals = np.real(evals)
evecs = np.real(evecs)

dpp_sample = sample_LDPP.sample_exact_k(evals, evecs, 50)

print(dpp_sample)
print(Y[dpp_sample, 0], ", ", Y[dpp_sample, 1])

unif_sample = np.random.permutation(len(Y))[:len(dpp_sample)]

plt.title("dpp")
plt.scatter(Y[dpp_sample, 0],Y[dpp_sample, 1])
plt.show()
plt.title("uniform")
plt.scatter(Y[unif_sample, 0], Y[unif_sample, 1])
plt.show()