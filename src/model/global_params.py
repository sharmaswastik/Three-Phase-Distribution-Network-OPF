import numpy as np
from pyomo.environ import *
count = 0
   
def update_func(A_0, A, D_r, D_x):
    
    R_d = np.matmul(np.linalg.inv(A), D_r)#, np.linalg.inv(A.T))
    X_d = np.matmul(np.linalg.inv(A), D_x)#, np.linalg.inv(A.T))
    A_dash = np.matmul(np.linalg.inv(A), A_0)
    
    return A_dash, R_d, X_d

def initialize_param():

    global count
    count = 0