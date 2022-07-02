# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 21:02:33 2022

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

def height(x,y):
    return ((x-25)**3 + (x-25) * (y-25) + (y-25)**3) * 0.01

def jacobian(x,y):
    """ For now working in a simple case, only interested in 
    Jacobian near two points x and y 
    
    Note: I think we actually need the cross product here
    (think surface area element from VC)
    """
    epsilon = 1
    bias = 100
    dx = (height(x + epsilon, y) - height(x,y)) / epsilon
    dy = (height(x, y + epsilon) - height(x,y)) / epsilon
    return (np.sqrt(dx**2 + dy**2 + 1) * epsilon**2) / bias + 1
    #return dx * dy + 1
    #return 1


""" Construct the L Kernel """

""" Recall Gram Decomposition of L Kernel into a matrix
L = B^T B 
Giving the quality-diversity decomposition
q_i phi_i^T phi_i q_i 

We take a coordinate phi_i and set q_i = Jacobian(evaluated at phi_i)

Only necessary to find B matrix for now, this case is low dimensional
so can easily multiply to find L = B^T B

L_ij = q_i S_ij q_j 
S(x, y) = exp(-1/2s^2 |x-y|^2)
N = # grid points 
dim = space dimension 
Y = ground set 
Q = SET of quality factors
L = L kernel 

Here we set the quality coefficients to all be 1. 
Only care about diversity model for this test.
"""

N = 50
dim = 2
sigma = 15
Y = np.zeros(shape=(N**2, dim))
Q = np.ones(shape=(N**2))
index = 0
for i in range(0, N):
    for j in range(0, N):
        Y[index, :] = [i, j]
        Q[index] = jacobian(i,j)
        index += 1

L = np.zeros(shape=(N**2, N**2))
for i in range(0, N**2):
    for j in range(0, N**2):
        
        L[i][j] = Q[i] * np.exp(-1/(2 * sigma) * 
                                np.linalg.norm(Y[i][:] 
                                               - Y[j][:])**2) * Q[j]
        """
        
        L[i][j] = np.exp(-1/(2 * sigma) * 
                                np.linalg.norm(Y[i][:] 
                                               - Y[j][:])**2)
        """
evals, evecs = np.linalg.eig(L)
evals = np.real(evals)
evecs = np.real(evecs)

dpp_sample = sample_LDPP.sample_exact_k(evals, evecs, 30)


print(dpp_sample)

plt.title("Planar Projection of DPP Model")
plt.scatter(Y[dpp_sample, 0],Y[dpp_sample, 1])
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')


ax.scatter3D(Y[dpp_sample, 0],Y[dpp_sample, 1],
             height(Y[dpp_sample, 0], Y[dpp_sample, 1]))

# Plot the energy landscape
x = np.linspace(0, 50, 100)
y = np.linspace(0, 50, 100)

A, B = np.meshgrid(x, y)
Z = height(A, B)
#ax.contour3D(A, B, Z, 500, cmap='binary')
ax.plot_wireframe(A, B, Z, rstride=10, cstride=10, color="black",
                  linewidth=0.2)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.grid(False)
plt.title("DPP Model")
plt.show()

unif_sample = np.random.permutation(len(Y))[:len(dpp_sample)]

fig = plt.figure()
ax = plt.axes(projection='3d')


ax.scatter3D(Y[unif_sample, 0],Y[unif_sample, 1],
             height(Y[unif_sample, 0], Y[unif_sample, 1]))

# Plot the energy landscape
x = np.linspace(0, 50, 100)
y = np.linspace(0, 50, 100)

A, B = np.meshgrid(x, y)
Z = height(A, B)
#ax.contour3D(A, B, Z, 500, cmap='binary')
ax.plot_wireframe(A, B, Z, rstride=10, cstride=10, color="black",
                  linewidth=0.2)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.grid(False)
plt.title("Naive Uniform Points on Surface")
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')

# Plot the quality factors
x = np.linspace(0, 50, 100)
y = np.linspace(0, 50, 100)

A, B = np.meshgrid(x, y)
Z = jacobian(A, B)
#ax.contour3D(A, B, Z, 500, cmap='binary')
ax.plot_wireframe(A, B, Z, rstride=10, cstride=10, color="black",
                  linewidth=0.5)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.grid(False)
plt.title("Quality Coefficients as Functions of Points")
plt.show()
