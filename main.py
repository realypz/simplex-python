# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 09:39:27 2021

@author: realy
"""

import numpy as np
from util import Pivoting, Simplex

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
    
    # [Linear program standard form]
    
    c = np.array([1., -0.5, 2.4, 1.4, 0.48], dtype=np.float)
    A = np.array([[5, 3, 5, 7.0, 0.32],
                   [1, 3, -2, 1.4, 2.1],
                   [1, 0, 0, 6, 1.6]], dtype=np.float)    
    b = np.array([2.2, 2.1, 1], dtype=np.float)
    
    Ab = np.hstack([A, b.reshape(-1,1)])        
    
    # Pivoting the Ab matrix
    pivot = Pivoting()
    Ab_ = pivot(Ab)
    
    
    # Make all b elements >= 0
    for i in range(Ab_.shape[0]):
        if Ab_[i,-1] < 0:
            Ab_[i,:] = -Ab_[i,:]
            
    A_ = Ab_[:,:-1]
    b_ = Ab_[:,-1]


    simplex = Simplex(c=c, A=A_, b=b_)
    simplex.alg()

    
    pass
    
    
    

    