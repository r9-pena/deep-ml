import numpy as np

def impute_missing_data(data: np.ndarray, strategy: str = 'mean') -> np.ndarray:
    """
    Impute missing values in a 2D array using the specified strategy.
    
    Args:
        data: 2D numpy array with missing values represented as np.nan
        strategy: Imputation strategy - 'mean', 'median', or 'mode'
        
    Returns:
        2D numpy array with missing values imputed
    """
    # Your code here
    x = data.copy()
    n,d = np.shape(x)
    mean = np.nanmean(x,axis=0)
    mean[mean==None]=0.0
    median = np.nanmedian(x,axis=0)
    mode = []
    for col in x.T:
        valid = col[~np.isnan(col)]
        values,counts = np.unique(valid, return_counts=True)
        mode.append(values[np.argmax(counts)])
    mode = np.asarray(mode)
    strg = {
        'mean':mean,
        'median':median,
        'mode':mode,
    }
    row,col = np.where(np.isnan(x))
    x[row,col] = strg[strategy][col]
    return x