# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 18:16:18 2022

@author: Maria
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import eigh

"""
Algorithm: Basin Hopping
Generate a random initial state Y_0 in D
X_0 = LocalMinimisation(Y_0)
n = 0
while {stopping criterion is not satisfied} do
    Y_n = RandomPerturbation(X_n)                       # Or DPP()
    U_n = LocalMinimisation(Y_n)
    Generate V ~ Uniform(0,1)
    if V < min( 1, exp(- (f(U_n) - f(X_n)) / T) ) then
        X_{n+1} = U_n
    else
        X_{n+1} = X_n
    end
    Increase n by 1
end
"""

class Potential:
    def __init__(self):
        pass
    
    def get_energy(self, x):
        return ( (x+2) * np.sin((x+2)) + 2*(x+2)) ** 2
    
    def get_grad(self, x):
        return 0

class AcceptTest:
    def __init__(self, T):
        self.T = T
    
    def Metropolis(self, coords, trial_coords):
        V = np.random.uniform(0, 1)
        criterion = np.exp(-(coords - trial_coords) / self.T)
        if V < min(1, criterion):
            return True
        else:
            return False
        
    def Strict(self, coords, trial_coords):
        if trial_coords < coords:
            return True
        else:
            return False
    

class DPPBasinHopping:
    
    def __init__(self, L, x0, potential, n_iter, DPP_coordinates, 
                 accept_test_type="Strict", optimizer_type="L-BFGS-B", verbose=0):
        """
        L - DPP L kernel
        x_0 - numpy array of initial coordinates
        potential - energy landscape function
        DPP_coordinates - the list of coordinates over which the determinantal
        point process will operate (should equal evals.shape[0])
        """
        
        """ BasinHopping/EnergyLandscape """
        self.potential = potential
        self.n_iter = n_iter
        self.accept_test = AcceptTest(5)
        self.accept_test_type = accept_test_type
        self.optimizer_type = optimizer_type
        
        # Initialising starting states, etc
        self.Y_n = x0
        local_minimisation = minimize(self.potential.get_energy, self.Y_n, method=self.optimizer_type)
        self.X_n = local_minimisation.x[0]
        
        if verbose == 1:
            print("Initial state", self.X_n)
        
        """ Points/Labels Mapping """
        labels = range(len(DPP_coordinates))
        self.points_map = dict(zip(labels, DPP_coordinates))
        
        """ DPP """
        #self.rng = np.random.default_rng()
        
        """ Sampling phase 1
        Calculate the 'inclusion probabilities'/'mixture weight'
        for each eigenvalue by
        mixture weight = lambda_i / (lambda_i + 1)
        Construct corresponding set of eigenvectors by restricting the
        set of eigenvectors to the set indexed by the chosen eigenvalues 
        """
        
        if verbose != 0:
            print("Computing eigendecomposition of L")
        
        evals, evecs = eigh(L)
        # ??? transpose?
        
           
        # Define the ground set size
        self.N = evals.shape[0] # alt. = len(DPP_coordinates)
        self.inclusion_probabilities = evals / (1 + evals)
        # Construct corresponding set of eigenvectors
        self.V = evecs[:, np.random.rand(self.N) < self.inclusion_probabilities]
        
        self.dim_V = self.V.shape[1]
        #Hold output
        self.Y = []
        self.i = self.dim_V - 1
        
        """ Other """
        self.verbose = verbose
    
    def gen_dpp_step(self):
        """ DPP step taking algorithm """
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
        
        if self.i > -1:
            # We are still able to sample
            
            # Create the sampling distribution
            P = np.sum(self.V**2, axis=1)
            P = P / np.sum(P)
            
            # Choose an element from the ground set according to probabilities
            item = np.random.choice(range(self.N), p=P)
            
            # Get first nonzero vector, find orthogonal subspace
            index = np.nonzero(self.V[item])[0][0]
            self.Y.append(item)
            
            # Update V
            V_j = np.copy(self.V[:, index])
            self.V = self.V - np.outer(V_j, self.V[item] / V_j[item])
            self.V[:, index] = self.V[:, self.i]
            self.V = self.V[:, :self.i]
            
            # Orthogonalise V by QR decomposition
            self.V, _ = np.linalg.qr(self.V)
            
            self.i -= 1
            y = self.Y[-1]
            
            # When y in Y is selected, use the points map
            coordinate = self.points_map[y]
            
            
            if self.verbose != 0:
                print("Taking a step!", coordinate)
                print("Current DPP point selection: ", coordinate)
            return coordinate
        else:
            print("No remaining dimensions of subspace")
            # 0 acts as a default point
            return 0
    
    def take_step(self):
        
        accept_step = 0
        
        """ Take step """
        
        self.Y_n = self.gen_dpp_step()
        
        """ Local optimisation """
        local_minimisation = minimize(self.potential.get_energy, self.Y_n, method=self.optimizer_type)
        U_n = local_minimisation.x[0]
        
        """ Check if step is a valid configuration """
        
        
        
        """ Accept testing """
        
        trial_coords = self.potential.get_energy(U_n)
        coords = self.potential.get_energy(self.X_n)
        
        if self.accept_test_type == "Strict":
            
            if self.accept_test.Strict(coords, trial_coords):
                self.X_n = U_n
                accept_step = 1
            else:
                pass
            
        elif self.accept_test_type == "Metropolis":
            
            if self.accept_test.Metropolis(coords, trial_coords):
                self.X_n = U_n
                accept_step = 1
            else:
                pass
            
        else:
            print("No such accept test, forcing no acceptance")
        
        
        if self.verbose != 0:
            print("State ", self.X_n, "Accepted ", accept_step)
            print("Trial coordinates, ", trial_coords, " original state coordinates, ",
                  coords)
    
    def run(self):
        for i in range(0, self.n_iter):
            self.take_step()
        
        return self.X_n


@np.vectorize
def func(x):
        return ( (x+2) * np.sin((x+2)) + 2*(x+2)) ** 2
    

if __name__ == "__main__":
    print("Python DPP-BasinHopping")
    
    n_coords = 40
    sigma = 1
    dpp_coords = np.linspace(-20, 20, n_coords).reshape(n_coords, 1)
    L = np.zeros(shape=(n_coords, n_coords))
    
    vals = func(dpp_coords.reshape(40))
    plt.plot(dpp_coords.reshape(40), vals)
    plt.show()
    
    for i in range(0, len(dpp_coords)):
        for j in range(0, len(dpp_coords)):
            L[i][j] = np.exp(-1 / (2 * (sigma) * np.linalg.norm(dpp_coords[i] - dpp_coords[j])**2))

    
    potential = Potential()
    dppbh = DPPBasinHopping(L, 14, potential, 10, dpp_coords, verbose=1)
    
    X = dppbh.run()
    print("Global Minima: ", X, potential.get_energy(X))