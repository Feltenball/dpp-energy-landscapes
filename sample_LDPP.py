# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 21:20:48 2022

@author: Maria
"""

import numpy as np


""" To do:
Implement sample_exact_k(evals, evecs, k)
"""

def sample_exact(evals, evecs):
    """ evals, evecs - eigendecomposition of L kernel.
    Samples an L DPP according to Algorithm 1 in Kuszela and Taskar
    (spectral decomposition method)
    """
    
    """ Sampling phase 1
    Calculate the 'inclusion probabilities'/'mixture weight'
    for each eigenvalue by
    mixture weight = lambda_i / (lambda_i + 1)
    Construct corresponding set of eigenvectors by restricting the
    set of eigenvectors to the set indexed by the chosen eigenvalues 
    """
    # Define the ground set size
    N = evals.shape[0]
    inclusion_probabilities = evals / (1 + evals)
    # Construct corresponding set of eigenvectors
    V = evecs[:,np.random.rand(N) < inclusion_probabilities]
    dim_V = V.shape[1]
    
    # Hold output
    Y = []
    
    """ Sampling phase 2
    We calculate the probabilities of selecting each item from the ground
    set as in Kuszela and Taskar Algorithm 1, that is,
    P(i) = \sum_{v \in V} (v^T e_i)^2
    Choose some i. (item)
    Now find a vector to eliminate. (index) 
    To do this, find the index of
    some vector in V which has a nonzero component e_i.
    Obtain the subspace perpendicular to e_i and orthogonalise.
    Note that QR decomposition is more numerically stable than 
    Gram-Schmidt for this purpose, but the effect achieved is identical.
    """
    
    for i in range(dim_V-1, -1, -1):
        P = np.sum(V**2, axis = 1)
        P = P / np.sum(P)
        
        # Choose an element from the ground set according 
        # to the probabilities
        item = np.random.choice(range(N), p=P)
        # row_ids is a scalar
        
        # Get the first nonzero vector 
        # First vector which has a nonzero element in that row
        # We will find subspace orthogonal to this vector
        index = np.nonzero(V[item])[0][0]
        # Note that axis 0 of the V array corresponds to the rows
        # So we can ensure we will not get out of bounds error
        
        Y.append(item)

        # update V
        # V_j is the vector we don't like
        V_j = np.copy(V[:,index])
        V = V - np.outer(V_j, 
                         V[item]/V_j[item])
        V[:,index] = V[:,i]
        V = V[:,:i]

        # Orthogonalise by using qr decomposition
        V, _ = np.linalg.qr(V)

    return Y

def elementary_symmetric_polynomial(evals, k):
    """
    eigenvalues evals, kth elementary symmetric polynomial.
    Returns the kth elementary symmetric polynomial
    Note k and n are indexed starting at 1 mathematically, but 0 internally.
    according to a recursive algorithm.
    Note e[k,N] = e_k(lambda_1, ..., lambda_N)

    We only care about
    k \in {1, ..., N}
    N \in {1, ..., N}
    But we define the 0th index for the purposes of the base case in 
    recursion.
    Therefore we take e.shape = (k+1, N+1) to allow space for the 0 index
    and the 1, ..., N, 1, ..., k.
    """
    N = evals.shape[0]
    e = np.zeros(shape=(k+1,N+1))
    e[0, :] = 1
    e[1:k+1, 0] = 0

    for l in range(1, k+1):
        for n in range(1, N+1):
            e[l, n] = e[l, n-1] + evals[n-1] * e[l-1][n-1]
    
    #e = np.delete(e, 0, axis=(0))
    #e = np.delete(e, 0, axis=(1))
    
    return e

def sample_k_mixture_components(evals, k):
    """
    Compute the chosen mixture components for the k-DPP 
    (phase 1 of sampling)
    Return an array J of numbers corresponding to which evecs are chosen.
    Kuszela and Taskar Algorithm 8
    """
    """ First compute the symmetric polynomials """
    N = evals.shape[0]
    e = elementary_symmetric_polynomial(evals, k)
    # Output set
    J = []
    l = k
    
    for n in range(N, 1, -1):
        # Chosen all evecs
        if l == 0:
            break
        
        # Compute the marginals
        if n == l:
            marginal = 1
        else:
            marginal = evals[n-1] * (e[l-1,n-1] / e[l,n])
        
        # Note we use n - 1 to account for the 0-indexing
        if np.random.uniform(0,1) < marginal:
            J.append(n - 1)
            l = l - 1

    return J
        
def sample_exact_k(evals, evecs, k):
    """ evals, evecs - eigendecomposition of L kernel.
    Samples an L DPP according to Algorithm 1 in Kuszela and Taskar
    (spectral decomposition method)
    """
    
    """ Sampling phase 1
    Calculate the 'inclusion probabilities'/'mixture weight'
    for each eigenvalue by
    mixture weight = lambda_i / (lambda_i + 1)
    Construct corresponding set of eigenvectors by restricting the
    set of eigenvectors to the set indexed by the chosen eigenvalues 
    """
    # Define the ground set size
    N = evals.shape[0]
    # Construct corresponding set of eigenvectors
    J = sample_k_mixture_components(evals, k)
    V = evecs[:, J]
    dim_V = V.shape[1]
    
    # Hold output
    Y = []
    
    """ Sampling phase 2
    We calculate the probabilities of selecting each item from the ground
    set as in Kuszela and Taskar Algorithm 1, that is,
    P(i) = \sum_{v \in V} (v^T e_i)^2
    Choose some i. (item)
    Now find a vector to eliminate. (index) 
    To do this, find the index of
    some vector in V which has a nonzero component e_i.
    Obtain the subspace perpendicular to e_i and orthogonalise.
    Note that QR decomposition is more numerically stable than 
    Gram-Schmidt for this purpose, but the effect achieved is identical.
    """
    
    for i in range(dim_V-1, -1, -1):
        P = np.sum(V**2, axis = 1)
        P = P / np.sum(P)
        
        # Choose an element from the ground set according 
        # to the probabilities
        item = np.random.choice(range(N), p=P)
        # row_ids is a scalar
        
        # Get the first nonzero vector 
        # First vector which has a nonzero element in that row
        # We will find subspace orthogonal to this vector
        index = np.nonzero(V[item])[0][0]
        # Note that axis 0 of the V array corresponds to the rows
        # So we can ensure we will not get out of bounds error
        
        Y.append(item)

        # update V
        # V_j is the vector we don't like
        V_j = np.copy(V[:,index])
        V = V - np.outer(V_j, 
                         V[item]/V_j[item])
        V[:,index] = V[:,i]
        V = V[:,:i]

        # Orthogonalise by using qr decomposition
        V, _ = np.linalg.qr(V)

    return Y
