import numpy as np

def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
    # Your code here
    x = data.copy()
    n,d = np.shape(x)
    mean = np.mean(x,axis=0)
    std = np.std(x,axis=0)
    min_col = np.min(x,axis=0)
    max_col = np.max(x,axis=0)
    standardized_data = (x-mean)/std
    normalized_data = (x-min_col)/(max_col-min_col)
    return standardized_data, normalized_data