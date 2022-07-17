# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 10:04:50 2022

@author: Maria
"""

""" DPPs Basin Hopping """

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import basinhopping
import time

b = 10
n_iter = 1000
bounds = np.linspace(-b, b, 50)


@np.vectorize
def f(x):
    return ( x * np.sin(x) + 2*x) ** 2

func = f(bounds)

""" Set up basinhopping 
- Add a callback function which prints minimums we progressively find
- Put bounds on the problem

"""

# Callback
def print_func(x, f, accepted):
        print("at minimum %.4f accepted %d" % (f, int(accepted)))
        
def print_func_and_return_minima(x, f, accepted):
        print("at minimum %.4f accepted %d" % (f, int(accepted)))
        return f

class Bounds:
    def __init__(self, x_max, x_min):
        self.x_max = np.array(x_max)
        self.x_min = np.array(x_min)
    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        t_max = bool(np.all(x <= self.x_max))
        t_min = bool(np.all(x >= self.x_min))
        return t_max and t_min



""" No DPPs """

"""
x0 = -9
minimizer_kwargs = {"method": "BFGS"}
ret = basinhopping(f, x0, minimizer_kwargs=minimizer_kwargs, niter=n_iter)

print("global minimum: x = %.4f, f(x0) = %.4f" % (ret.x, ret.fun))
global_min = ret.x

plt.plot(bounds, func)
plt.scatter(ret.x, f(ret.x), color="green", linewidth=0.5)
plt.show()




basinhopping_bounds = Bounds(-b, b)

ret = basinhopping(f, x0, minimizer_kwargs=minimizer_kwargs,
                   niter=n_iter, callback=print_func,
                   accept_test = basinhopping_bounds)
print("global minimum: x = %.4f, f(x0) = %.4f" % (ret.x, ret.fun))
"""


""" Convergence Plots """

"""
n_experiments = 1000
outcomes = np.zeros(n_experiments)

for n in range(1, n_experiments):
    x_0 = np.random.uniform(-10, 10)
    
    ret = basinhopping(f, x_0, minimizer_kwargs=minimizer_kwargs,
                   niter=n,
                   accept_test = basinhopping_bounds)
    
    outcomes[n] = np.abs(f(ret.x))

plt.plot(outcomes)
plt.show()
"""

"""

Design an alternate step taking algorithm that works in conjunction
with a determinantal point process.


"""

class StepTaking:
   def __init__(self, stepsize=0.5):
       self.stepsize = stepsize
       self.rng = np.random.default_rng()
   def __call__(self, x):
       s = self.stepsize
       x[0] += self.rng.uniform(-2.*s, 2.*s)
       x[1:] += self.rng.uniform(-s, s, x[1:].shape)
       return x
   
