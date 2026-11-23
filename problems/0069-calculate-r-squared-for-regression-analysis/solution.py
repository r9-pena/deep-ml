
import numpy as np

def r_squared(y_true, y_pred):
    mean = np.mean(y_true)
    guess_error = np.sum((y_true - mean)**2)
    error = np.sum((y_true-y_pred)**2)
    return (guess_error-error)/guess_error