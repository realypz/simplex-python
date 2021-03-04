# -*- coding: utf-8 -*-
import sys
import numpy as np
import sympy
from itertools import combinations, permutations


def LoadSingleTestExample(input):
    c, A, b = input["c"], input["A"], input["b"]
    n_variables = len(c)
    for constraint_coeff in A:
        assert len(constraint_coeff) == n_variables
    assert len(b) == len(A)
    return c, A, b


def ReducedRowEchelonForm(nparr):
    r"""
    Return the "Reduced Row Echelon Form" of an input array.

    Parameters
    ----------
    nparr : np.array

    Returns
    -------
    np.array
        The Reduced Row Echelon Form of input.

    """
    rref_sym = sympy.Matrix(nparr).rref()[0].tolist()
    nv, nh = nparr.shape
    rref_val = []
    
    for i in range(nv):
        tmp = []
        for j in range(nh):
            rref_sym_ij = rref_sym[i][j]
            tmp.append(rref_sym_ij.p / rref_sym_ij.q)
        rref_val.append(tmp)
    
    return np.array(rref_val)


class Pivoting:
    def __init__(self):
        pass
        
    def __call__(self, nparr):
        self.nparr = nparr.copy()
        self.nparr_cpy = nparr
        self.nrow, self.ncol = np.shape(self.nparr)
        after_pivoting = self.run()
        return after_pivoting
        
    def run(self):   
        nrow, ncol = np.shape(self.nparr)       
        newRowID = self._rowsort()

        rowptr = 1
        
        if (rowptr == nrow):
            return 
        
        while (rowptr < nrow):
            cur_rowID = newRowID[rowptr]
            cur_row = self.nparr[cur_rowID].copy()
            
            last_rowID = newRowID[rowptr - 1]
            last_row = self.nparr[last_rowID].copy()
            
            colID = self._GetFirstNonZeroIdx(cur_row)
            
            if (colID < rowptr):     
                cur_row = cur_row - (cur_row[colID] / last_row[colID]) * last_row
                self.nparr[cur_rowID] = cur_row             
                # Resort
                newRowID = self._rowsort()
                pass
            
            else:
                pass
            
            if (self._GetFirstNonZeroIdx(self.nparr[newRowID[rowptr]]) > self._GetFirstNonZeroIdx(self.nparr[newRowID[rowptr-1]])):
                rowptr += 1
            
        return self.nparr[newRowID]
        

    def _rowsort(self):
        firstNoneZeroIndice = np.zeros((self.nrow), dtype=np.int64)
        
        for rowId in range(self.nrow):
            cur_row = self.nparr[rowId]
            firstNoneZeroIndice[rowId] = self._GetFirstNonZeroIdx(cur_row)
        
        newColOrder = np.argsort(firstNoneZeroIndice)
        return newColOrder
  
    
    def _GetFirstNonZeroIdx(self, row):
        assert row.ndim == 1
        first_nonzero_idx = 0
        ncol = row.shape[0]
        
        for i in range(ncol):
            if (row[first_nonzero_idx]) == 0:
                first_nonzero_idx += 1
            else:
                break
        
        return first_nonzero_idx
    

class SimplexCore():    
    def __init__(self):
        pass

    def __call__(self, c, A, b):
        self.init(c, A, b)
        self.alg()

    def init(self, c, A, b):
        self.c = np.asmatrix(c).reshape(-1,1)
        self.A = np.asmatrix(A)
        self.b = np.asmatrix(b).reshape(-1,1)
        
        # Todo: Check c, A, b is the standard form of a linear program
        
        self.m_constraints = self.A.shape[0]
        self.n_variables = self.A.shape[1]
        
        self.VARIABLES = np.arange(0, self.n_variables, 1)

        # self.solution_status must be one of ["not started", "successful", "unbounded", "infeasible"]
        self.solution_status = "not started"
        
        self.solution = None
        self.optimal_value = None
        self.foundBFS = None

        ## Some constants
        # The floating computation error allowance
        self.EPS = 1e-7
        
        #print("======== Standform =========\n")
        #print("c.T")
        #print(self.c)
        #print("\n")
        
        #print("self.A")
        #print(self.A)
        #print("\n")
        
        #print("self.b")
        #print(self.b)
        #print("\n")
    
    def alg(self):               
        basic_variables, nonbasic_variables = self.find_BFS()
        if not self.foundBFS:
            return
        x_solution = np.asmatrix(np.zeros((self.n_variables,1), dtype=np.float))
        
        while(True):
            B = self.A[:, basic_variables]
            N = self.A[:, nonbasic_variables]
            
            # Calculate reduced cost
            B_inv = np.linalg.inv(B)
            reduced_cost = self.c[nonbasic_variables,:].T - self.c[basic_variables,:].T @ B_inv @ N
            
            if (reduced_cost < -self.EPS).sum() == 0:
                # all the reduced cost is non-negative, this means we reach the optimal
                x_basic = B_inv @ self.b
                x_solution[basic_variables,:] = x_basic
                self.solution = x_solution.reshape(-1,).tolist()
                self.solution_status = "successful"
                self.optimal_value = np.asscalar(self.c.T @ x_solution)
                break
            
            else:
                # not optimal, we need to either:
                # 1. decide whether the problem is unbounded or not. If unbounded, terminte the while loop.
                # 2. exchange an variable between basic and non-basic
                                
                # decide which variable to enter the basis
                new_basis = nonbasic_variables[np.argmin(reduced_cost)]
                
                # Is the problem unbounded?
                if self.DecideUnboundness(B_inv, self.A[:,new_basis]) == True:
                    self.solution_status = "unbounded"
                    print("Problem unbounded. algorithm terminated...")
                    break
                else: # The prblem is boundedm then find a basic variable to unbounded... 
                    x_basic = B_inv @ self.b
                    Binv_Nj = B_inv @ self.A[:,new_basis]
                    
                    Binv_Nj_negative_mask = np.asarray(Binv_Nj < self.EPS).reshape(-1,)
                    
                    ratio = x_basic / Binv_Nj
                    ratio[Binv_Nj_negative_mask,:] = np.Inf              
                    variable_to_leave_basis = basic_variables[np.argmin(ratio)]

                    basic_variables = np.setdiff1d(basic_variables, variable_to_leave_basis)
                    basic_variables = np.union1d(basic_variables, new_basis)
                    
                    nonbasic_variables = np.setdiff1d(nonbasic_variables, new_basis)
                    nonbasic_variables = np.union1d(nonbasic_variables, variable_to_leave_basis)

    
    def DecideUnboundness(self, B_inv, Nj):
        tmp = B_inv @ Nj
        mask = tmp < -self.EPS
        
        if mask.sum() == self.m_constraints:
            return True  # unbounded
        else:
            return False # bounded
        
 
    def find_BFS(self):
        foundBFS = False
        basic_variables_all = list(combinations(np.arange(0, self.n_variables, 1), self.m_constraints))
        
        for basic_variables in basic_variables_all:
            nonbasic_variables = np.setdiff1d(self.VARIABLES, basic_variables)
            B = self.A[:, basic_variables]
            B_inv = np.linalg.inv(B) 
            x_basic = B_inv @ self.b
            
            print("(x_basic > 0).sum() =", (x_basic > 0).sum())
            if (x_basic > 0).sum() == self.m_constraints:
                foundBFS = True
                break
            
        if foundBFS == True:
            self.foundBFS = True
            return basic_variables, nonbasic_variables
        else:
            self.foundBFS = False
            self.solution_status = "infeasible"
            print("This problem is infeasible. Program exit.\n")
            return None, None
                
            
    
class Simplex():
    def __init__(self):
        self.pivoting = Pivoting()
        self.simplex_core = SimplexCore()

    def __call__(self, c, A, b):
        self.c = c
        self.A = A
        self.b = b

        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float).reshape(-1,1)
        Ab = np.hstack([A, b])        
    
        # Pivoting the Ab matrix
        Ab = self.pivoting(Ab)
        # Make all b elements >= 0
        for i in range(Ab.shape[0]):
            if Ab[i,-1] < 0:
                Ab[i,:] = -Ab[i,:]

        A = Ab[:,:-1]
        b = Ab[:,-1]
        
        self.simplex_core(c=c, A=A, b=b)
        
    
    
    
    
    
    
    
    