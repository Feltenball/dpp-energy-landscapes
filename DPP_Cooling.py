# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt 
import sample_LDPP as ldpp
import scipy
from scipy.spatial.distance import pdist, squareform

"""

Comments

Strong effect of data transformations (frequency of maxima)
"spikiness" of quality scores


"""

""" First I'll define my curve """

bounds = np.linspace(-70, 70, 250)
frequency = 0.1
curve = bounds *np.sin(frequency * bounds) / 10
grad = ( np.sin(frequency * bounds) + frequency * bounds * np.cos(frequency * bounds) ) / 10
#curve = np.exp(- bounds**2)
#grad = - bounds * np.exp(- bounds**2)
sigma = 20
b_0 = 0
k = 10

""" Creating the DPP kernel 
We will introduce a hyperparameter which will be altered to vary repulsion
This hyperparameter will be altered during the cooling process to alter
local repulsion
So the feature vectors in the kernel consist of n-dimensional points
+ a hyperparameter corresponding to temperature
"""

features = np.ones(shape=(len(bounds), 2))
q = np.ones(shape=(len(bounds)))
q = np.exp(- grad**2 / 2) * 10
for i in range(0, len(features)):
    features[i][0] = bounds[i]
    features[i][1] = b_0

S= np.zeros(shape=(len(features), len(features)))
for i in range(0, len(bounds)):
    for j in range(0, len(bounds)):
        S[i][j] = q[i]  * np.exp(-np.linalg.norm(features[i] - features[j])**2 / (2 * sigma**2)) * q[j]

evals, evecs = np.linalg.eig(S)
evals = np.real(evals)
evecs = np.real(evecs)

sample = ldpp.sample_exact(evals, evecs)
#sample = ldpp.sample_exact_k(evals, evecs, 6)

print(bounds[sample])

plt.suptitle("Frequency of xsin(wx): {}".format(frequency))

plt.subplot(3, 2, 1)
plt.title("Sample")
plt.plot(bounds, curve)
plt.scatter(bounds[sample], curve[sample], color="red")

plt.subplot(3, 2, 2)
plt.title("Gradient")
plt.plot(bounds, grad)

plt.subplot(3, 2, 3)
plt.title("Quality")
plt.plot(bounds, q)
plt.show()
"""
Now can we alter the hyperparameter to "cool down" the DPP iteratively
That is, where the gradient is lower, we want to make those points higher
quality and less repulsive
and where the gradient is higher we want to make those points more
repulsive
For now let's just use a Gaussian function.
Grad increases ---> higher repulsion. Can try arctan
"""

"""
n_iter = 500

for i in range(0, n_iter):
    evals, evecs = np.linalg.eig(S)
    evals = np.real(evals)
    evecs = np.real(evecs)
    sample = ldpp.sample_exact(evals, evecs)
    score = grad[sample]
    quality = np.exp(-score**2 / 2)
    #diversity = np.arctan(score)

    for i in range(0, len(sample)):
        for j in range(0, len(sample)):
            q[sample[i]] = quality[i]
            q[sample[j]] = quality[j]
            S[sample[i]][sample[j]] = q[i] * np.exp(
                -np.linalg.norm(features[sample[i]] - features[sample[j]])**2 
                / (2 * sigma**2)) * q[j]

plt.plot(bounds, curve)
plt.plot(bounds, grad, color="green")
#plt.scatter(bounds[sample], q[:], color="green")
plt.scatter(bounds[sample], curve[sample], color="red")
plt.show()

plt.plot(bounds, q)
"""

""" DPP Cooling """
"""
bounds = np.linspace(-10, 10, 250)
curve = bounds * np.sin(bounds)/ 10
sigma = 20
k = 10

plt.plot(bounds, curve)


# Create a DPP ontop of the landscape
# Similarity matrix will just be a Gaussian kernel for now
S = np.zeros(shape=(len(bounds), len(bounds)))

for i in range(0, len(bounds)):
    for j in range(0, len(bounds)):
        S[i][j] = np.exp(-np.abs(bounds[i] - bounds[j])**2 / (2 * sigma**2))
        
        
        q_i = np.abs(curve[i])
        q_j = np.abs(curve[j])
        S[i][j] = q_i * S[i][j] * q_j
        
        
evals, evecs = np.linalg.eig(S)
evals = np.real(evals)
evecs = np.real(evecs)

sample = ldpp.sample_exact_k(evals, evecs, k)

print(bounds[sample])

plt.scatter(bounds[sample], curve[sample], color="red")
plt.show()

# Now can we resample using the information we've gained to make 
# better predictions?
# Items which were further away from the minima should begin to gain
# repulsion on successive iterations
"""
""" Concept

Sample from DPP
Update quality model by making unwanted points exhibit more repulsion
Recompute the kernel and recompute the eigendecomposition
repeat

"""
"""
n_iter = 20
grad = ( bounds + bounds * np.cos(bounds) ) / 10

for i in range(0, n_iter):
    evals, evecs = np.linalg.eig(S)
    evals = np.real(evals)
    evecs = np.real(evecs)
    sample = ldpp.sample_exact(evals, evecs)
    #print(sample)
    score = grad[sample]
    q = np.exp(-score**2 / 2) / 100
    for i in range(0, len(sample)):
        for j in range(0, len(sample)):
            S[sample[i]][sample[j]] = q[i] * np.exp(
                -np.abs(bounds[sample[i]] - bounds[sample[j]])**2 
                / (2 * sigma**2)) * q[j]
    
    plt.plot(bounds, curve)
    plt.scatter(bounds[sample], q[:], color="green")
    plt.scatter(bounds[sample], curve[sample], color="red")
    plt.show()
    
plt.plot(bounds, grad)
plt.show()
plt.plot(bounds, 1 * np.exp(-grad**2 / 2), color="red")
"""