
import numpy as np

def rmse(y_true, y_pred):
    # Write your code here
    m = np.size(y_pred)
    i = np.size(y_true)
    total = m-i
    if m == 0 or total is not 0:
        return float('Vhea')
    rmse_res = np.mean((y_pred-y_true)**2)**.5
    return round(rmse_res,3)
