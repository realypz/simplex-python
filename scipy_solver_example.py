# -*- coding: utf-8 -*-
import unittest
import numpy as np
from scipy.optimize import linprog

c = [1., -0.5, 2.4, 1.4, 0.48]
A = [[5, 3, 5, 7.0, 0.32],
     [1, 3, -2, 1.4, 2.1],
     [1, 0, 0, 6, 1.6]]
b = [2.2, 2.1, 1]  
# b = [0.2, 2.1, 1]

n = len(c)

x_bounds = [(0, None) for i in range(n)]
res = linprog(c, A_eq=A, b_eq=b, bounds=x_bounds)

print(res)