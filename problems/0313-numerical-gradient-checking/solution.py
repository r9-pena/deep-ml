import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    """
    Perform numerical gradient checking using centered finite differences.
    
    Args:
        f: A function that takes a numpy array and returns a scalar
        x: numpy array, the point at which to check gradient
        analytical_grad: numpy array, the analytically computed gradient
        epsilon: float, small value for finite difference approximation
    
    Returns:
        tuple: (numerical_grad, relative_error)
    """
    # Your code here
    x = np.asarray(x,dtype=float)
    gn = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += epsilon
        x_minus[i] -= epsilon
        gn[i] = (f(x_plus)-f(x_minus))/(2*epsilon)

    denominator = (np.linalg.norm(gn)+np.linalg.norm(analytical_grad))
    error = np.linalg.norm(gn-analytical_grad)/denominator if denominator != 0 else 0.0
    return (gn,error)