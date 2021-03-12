# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 09:39:27 2021

@author: realy
"""

import numpy as np
import json
from util import Pivoting, SimplexCore, Simplex, LoadSingleTestExample

if __name__ == "__main__":

    r'''
    The standard form is
    minimize    y = c.T * x
    subject to        A * x = b
                          x >= 0(n)
                          
    A∈R(m x n), rank(A) = m.
    b >= 0(m)
    m < n
    x has "n" elements.
    '''
    
    with open("test_examples.json", "r") as f:
        test_data = json.load(f)
    
    standard_problem = test_data[5]
    c, A, b = LoadSingleTestExample(standard_problem)
    
    simplex = Simplex()
    simplex(c, A, b)

    pass
    
    
    

    