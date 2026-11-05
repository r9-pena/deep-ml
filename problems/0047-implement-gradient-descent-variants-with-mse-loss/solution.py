import numpy as np

def batch_GD(X,y,weights,learning_rate,n_epochs):
    m,n = np.shape(X)
    for n in range(n_epochs):
        weights = weights - learning_rate*(2/m)*X.T.dot(X.dot(weights)-y)
    return weights

def SGD(X,y,weights,learning_rate,n_epochs):
    m,n = np.shape(X)
    for n in range(n_epochs):
        for i in range(m):
            weights = weights - learning_rate*(2)*X[i].T.dot(X[i].dot(weights)-y[i])
    return weights

def mini_batch(X,y,weights,learning_rate,n_epochs, batch_size):
    m,n = np.shape(X)
    slices = int(m/batch_size)
    x_batches = np.vsplit(X,slices)
    y_batches = np.split(y,slices)
    batch=0
    # return y_batches
    for epoch in range(n_epochs):
        for i in range(slices):
            weights = weights - learning_rate*(2/batch_size)*x_batches[i].T.dot(x_batches[i].dot(weights)-y_batches[i])
    return weights

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    if method == 'batch':
        return batch_GD(X,y,weights,learning_rate,n_epochs)
    elif method == 'stochastic':
        return SGD(X,y,weights,learning_rate,n_epochs)
    elif method == 'mini_batch':
        return mini_batch(X,y,weights,learning_rate,n_epochs,batch_size)
    pass
