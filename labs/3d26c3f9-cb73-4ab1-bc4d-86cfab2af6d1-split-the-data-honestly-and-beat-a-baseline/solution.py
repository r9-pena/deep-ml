# --- cell 1 ---
import numpy as np


def split_and_baseline(X, y, train_frac, val_frac, test_frac, seed):
    """Split rows into train/val/test folds, fit something on the training fold, predict the test fold.

    Returns
    -------
    test_predictions : np.ndarray, shape (len(test_idx),)
    train_idx, val_idx, test_idx : 1-D integer arrays forming a partition of range(len(y))
    """
    x = X.copy()
    n,d = np.shape(x)
    train_end = int(train_frac*n)
    val_end = train_end + int(val_frac*n)
    idx = np.random.default_rng(seed).permutation(n)
    train = idx[:train_end]
    validate = idx[train_end:val_end]
    test = idx[val_end:]
    test_predictions = y[idx[val_end:]]
    return test_predictions,train,validate,test