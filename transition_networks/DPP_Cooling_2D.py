# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sample_LDPP  as ldpp


""" Set up functions, variables, etc """

sigma = 2
x_bounds = np.linspace(-10, 10, 20)
y_bounds = np.linspace(-10, 10, 20)

def normalise(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

@np.vectorize
def height(x,y):
    #return ((x-25)**3 + (x-25) * (y-25) + (y-25)**3) * 0.01
    return np.exp( -((x-5)**2 + (y-5)**2) / 2 * sigma) - np.exp( -((x)**2 + (y)**2) / 2 * sigma) + np.exp( -((x+5)**2 + (y+5)**2) / 2 * sigma) 

"""
Q: Gradient is a vector, so how precisely should the quality
measures in the 1D case generalise
---> For now just work with the norm

Note that the sign of the norm is okay, since quality scores
will end up being in R+ anyway.
Although it's a bit dangerous to pass this into the Gaussian
function I'm using without a bit more care.
"""
@np.vectorize
def grad_exact(x,y):
    partial_x = (-(x - 5) / sigma) * np.exp( -((x-5)**2 + (y-5)**2) / 2 * sigma) + (x/sigma) * np.exp( -((x)**2 + (y)**2) / 2 * sigma) - ((x + 5) / sigma) * np.exp( -((x+5)**2 + (y+5)**2) / 2 * sigma)
    
    partial_y = (-(y - 5) / sigma) * np.exp( -((x-5)**2 + (y-5)**2) / 2 * sigma) + (y/sigma) * np.exp( -((x)**2 + (y)**2) / 2 * sigma) - ((y + 5) / sigma) * np.exp( -((x+5)**2 + (y+5)**2) / 2 * sigma)
    
    norm = np.linalg.norm([partial_x, partial_y])
    
    return norm

""" DPPs 

Similarity model. S_ij RBF Kernel.
The quality scores q_i as functions of features
GAUSSIAN of magnitude of grad
eg e^(-grad^2 / 2 * sigma) * lambda
"""

rescale = 1

N_features = len(x_bounds) * len(y_bounds)
features = np.ones(shape=(N_features, 2))
q = np.ones(shape=(N_features))

index= 0
for i in range(0, len(x_bounds)):
    for j in range(0, len(y_bounds)):
        features[index, :] = [x_bounds[i], y_bounds[j]]
        index += 1
        
q = np.exp( - grad_exact(features[:, 0], features[:, 1]) / (2 * sigma)) / rescale

# Set up the similarity matrix
S= np.zeros(shape=(len(features), len(features)))
for i in range(0, len(features)):
    for j in range(0, len(features)):
        S[i][j] = q[i]  * np.exp(-np.linalg.norm(features[i] - features[j])**2 / (2 * sigma**2)) * q[j]

evals, evecs = np.linalg.eig(S)
evals = np.real(evals)
evecs = np.real(evecs)

#dpp_sample = ldpp.sample_exact(evals, evecs)
dpp_sample = ldpp.sample_exact_k(evals, evecs, 10)


"""
PLOTTING
"""

fig = plt.figure()
ax = plt.axes(projection='3d')

# Plot the energy landscape
A, B = np.meshgrid(x_bounds, y_bounds)
Z = height(A, B)


ax.scatter3D(features[dpp_sample, 0],features[dpp_sample, 1],
             height(features[dpp_sample, 0], features[dpp_sample, 1]), color="blue")

ax.contour3D(A, B, Z, 500, cmap='binary')
#ax.plot_wireframe(A, B, Z, rstride=10, cstride=10, color="black",
                  #linewidth=1)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
              
ax.grid(False)

#plt.subplot(2, 2, 1)
plt.title("Energy Landscape")
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')

A, B = np.meshgrid(x_bounds, y_bounds)
Z = grad_exact(A, B)

ax.contour3D(A, B, Z, 500, cmap='binary')
#ax.plot_wireframe(A, B, Z, rstride=10, cstride=10, color="black",
                  #linewidth=1)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
              
ax.grid(False)

#plt.subplot(2, 2, 2)
plt.title("Magnitude Grad Field")
plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')

A, B = np.meshgrid(x_bounds, y_bounds)
Z = np.exp( - (grad_exact(A,B)**2) / (2 * sigma)) / rescale

ax.contour3D(A, B, Z, 500, cmap='binary')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.grid(False)

#plt.subplot(2, 2, 2)
plt.title("Quality Scores")
plt.show()


""" Contour Plots """

A, B = np.meshgrid(x_bounds, y_bounds)
Z = height(A, B)
fig, ax = plt.subplots(2)
cp = ax[0].contourf(A, B, Z)
fig.colorbar(cp) 
ax[0].set_title('Energy Landscape')

ax[0].plot()

Z = np.exp( - (grad_exact(A,B)**2) / (2 * sigma)) / rescale
cp = ax[1].contourf(A, B, Z)

ax[1].set_title('Quality Scores')

ax[1].plot()

plt.tight_layout(2, 2, 2)

plt.show()