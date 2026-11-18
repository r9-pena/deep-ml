import numpy as np

def random_split(data: np.ndarray, train_frac: float, validation_frac: float, seed: int = 123) -> list:
    """
    Randomly split a dataset into train, validation, and test subsets.
    """
    # Your code here
    n,d = np.shape(data)
    x = data.copy()
    train_end = int(train_frac*n)
    valid_end = train_end + int(validation_frac*n)
    idx = np.random.default_rng(seed).permutation(n)
    train = x[idx[:train_end],]
    valid = x[idx[train_end:valid_end],]
    test = x[idx[valid_end:],]
    return [train, valid, test]