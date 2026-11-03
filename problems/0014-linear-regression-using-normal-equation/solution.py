import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round
	x = np.array(X)
	y = np.array([y]).T
	nom = x.T.dot(y)
	den = x.T.dot(x)
	theta = np.linalg.inv(den).dot(nom)

	return theta.flatten()