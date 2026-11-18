import numpy as np

def to_categorical(x, n_col=None):
    # Your code here
    n_col = max(x)+1 if n_col is None else n_col
    idx = np.arange(n_col)
    encoded = (x[:,None] == idx[None,:]).astype(int)
    return encoded