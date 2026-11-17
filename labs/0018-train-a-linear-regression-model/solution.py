# --- cell 1 ---
import numpy as np

def train(X, y, W, b):
    """
    Train linear regression weights on standardized data.
    
    Args:
        X: numpy array of shape (m_samples, n_features) -- standardized features
        y: numpy array of shape (m_samples,) -- standardized targets
        W: numpy array of shape (n_features,) -- initial random weights
        b: float -- initial bias (0.0)
    
    Returns:
        W: numpy array of shape (n_features,) -- trained weights
        b: float -- trained bias
    """
    # TODO: implement your training strategy here
    # You can use ANY approach: gradient descent, normal equation,
    # momentum, adaptive learning rates, mini-batching, etc.
    m,n = np.shape(X)
    alpha = 0.1
    iterations = 100000
    for i in range(iterations):
        y_pred = X@W+b
        error = y_pred-y
        Loss = (1/m) * np.sum((error)**2)
        if Loss < 0.01:
            break
        dW = (2/m)*X.T@(error)
        db = 2*np.sum(error)/m
        W = W - alpha*dW
        b = b - alpha*db
    print(b)
    return W,b