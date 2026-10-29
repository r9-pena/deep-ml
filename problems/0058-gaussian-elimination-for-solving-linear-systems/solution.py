import numpy as np

def pivot(A_mod, col, total_rows):
    pivot = col
    for i in range(col,total_rows):
        if abs(A_mod[i][col]) > abs(A_mod[pivot][col]):
            pivot = i
    A_mod[[col, pivot]] = A_mod[[pivot,col]]
    for i in range(col+1,total_rows):
        scalar = A_mod[i][col]/A_mod[col][col]
        A_mod[i] = A_mod[i] - A_mod[col]*scalar
        # return scalar
    # return pivot
    return A_mod

def gaussian_elimination(A, b):
    """
    Solves the system Ax = b using Gaussian Elimination with partial pivoting.
    
    :param A: Coefficient matrix
    :param b: Right-hand side vector
    :return: Solution vector x
    """
    rows,cols = np.shape(A)
    A_mod = np.hstack((A,b.reshape(-1,1)))

    for i in range(cols):
        A_mod = pivot(A_mod,i,rows)
    # A_mod = pivot(A_mod,0,rows)
    a_new,b_new = np.hsplit(A_mod,np.array([rows]))
    res = np.linalg.inv(a_new).dot(b_new)
    # res = np.reshape(res, (1,-1))
    return res.flatten()