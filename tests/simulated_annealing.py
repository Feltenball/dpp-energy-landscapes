# -*- coding: utf-8 -*-

""" Simulated Annealing """

import numpy as np
import matplotlib.pyplot as plt

acceptance_criterions =[]
paths = []
paths2 = []

def f(x):
    return x[0]**2

def simulated_annealing(f, bounds, n_iter, step_size, temp):
    # Initial state
    best_point = bounds[:, 0] + np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    best_state = f(best_point)

    current_point, current_state = best_point, best_state
    for i in range(n_iter):
        candidate_point = current_point + np.random.randn(len(bounds)) * step_size
        
        candidate_state = f(candidate_point)
        
        if candidate_state < best_state:
            best_point, best_state = candidate_point, candidate_state
            paths.append(best_state)
            print(i, best_point, best_state)
            
        difference = candidate_state - current_state
        T = temp / float(i + 1)
        
        # Boltzmann Gibbs distrbution
        acceptance_criterion = np.math.exp(-difference / T)
        
        # For matplotlib plotting
        if difference > 0:
            acceptance_criterions.append(acceptance_criterion)

        if difference < 0 or np.random.rand() < acceptance_criterion:
            current_point, current_state = candidate_point, candidate_state
            paths2.append(current_state)
            
    return [best_point, best_state]

np.random.seed(1)

bounds = np.asarray([[-5, 5]])
best, score = simulated_annealing(f, bounds, 1000, 0.0000001, 10)
print(best, score)

plt.plot(acceptance_criterions)
plt.show()
plt.plot(paths)
plt.show()
plt.plot(paths2)
plt.show()