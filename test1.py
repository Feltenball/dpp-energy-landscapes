# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 16:32:43 2022

@author: Maria
"""

import numpy as np
import matplotlib.pyplot as plt
import sample_LDPP


def normalise(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

N = 50
k = 30
dim = 2
Y = np.zeros(shape=(N**2, dim))
index = 0
for i in range(0, N):
    for j in range(0, N):
        Y[index, :] = [i, j]
        index += 1

# Construct B matrix using columns of Y
# B is an DxN matrix (B^TB then has shape NxN)
B = np.zeros(shape=(dim + 3, N**2))
B[0:2, :] = np.copy(Y.T)

# Add a repulsion increasing hyperparameter
hyperparam = np.zeros(shape=(N**2))
hyperparam[:] = 100

B[-1][:] = hyperparam


for i in range(0, N**2):
    B[:, i] = 1-normalise(B[:, i])
    #B[i, :] = 1 * B[i, :] 

L = np.dot(B.T, B)

# Now generate a sample
# Prepare
# Eigendecomposition
evals, evecs = np.linalg.eig(L)
evals = np.real(evals)
evecs = np.real(evecs)

dpp_sample = sample_LDPP.sample_exact_k(evals, evecs, k)
#dpp_sample = sample_LDPP.sample_exact(evals, evecs)

print(dpp_sample)
print(Y[dpp_sample, 0], ", ", Y[dpp_sample, 1])

unif_smpl = np.random.permutation(len(Y))[:len(dpp_sample)]

plt.title("dpp cosine distance")
plt.scatter(Y[dpp_sample, 0],Y[dpp_sample, 1])
plt.show()
"""
plt.title("unif")
plt.scatter(Y[unif_smpl, 0], Y[unif_smpl, 1])
plt.show()
"""
#plt.imshow(L, cmap='hot', interpolation='nearest')
#plt.show()