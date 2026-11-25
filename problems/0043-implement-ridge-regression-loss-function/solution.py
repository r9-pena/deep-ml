import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
    l_std = np.mean((y_true-X@w.T)**2)
    ridge = alpha*w@w.T
    return l_std+ridge
