# -*- coding: utf-8 -*-

import numpy as np
import sympy

from util import ReducedRowEchelonForm, Pivoting

if __name__ == "__main__":
    A = np.array([[0,0,4,-4], [4,6,-3,0], [0,4,2,-3]])

    A_rref = ReducedRowEchelonForm(A)

    # print(A_rref)
     
    B0 = np.array([[0, 0, -3, 2, 0],
                  [-1, 4, 4, 7, 0],
                  [0 ,0, -76, 2.3, 2.08],
                  [9, 3, 2, -98.3, 2.34],
                  [0,0,0,0,3.4]], dtype=float)
    
    B1 = np.array([[0, 0, -3, 2, 0],
                  [-1, 0, 0, 7, 4],
                  [0 ,0, 9, -6, 0],
                  [0, 3, 2, -98.3, 2.34],
                  [0,3,0,0,3.4]], dtype=float)
    
    B2 = np.array([[1.2, 4.5, -0.9, 0.72, 0.09, -1.47],
                   [1.2, 0.0, 0.0, 0.0, 3.4, -2.45],
                   [0, 0, 3.4, -2.1, 3.7, 0.986],
                   [0, -0.98, 1.45, 0, 0, 0],
                   [4.3, -0.98, 3.1, 3.3, 0, 0]], dtype=float)
    
    B3 = np.array([[5,3,5,7],
                   [1,3,4,0],
                   [1,0,0,6]], dtype=float) # must specify dtype
    
    B = B3
    
    pivot = Pivoting()
    C = pivot(B)
    print(C)
    
    y = np.dot(B3[:,:-1], B3[:,-1])
    
