def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
	import numpy as np
	a = np.array(a)
	b = np.array(b)
	a_shape = np.shape(a)
	b_shape = np.shape(b)
	# row_a = np.shape(a)[0]
	# col_a = np.shape(a)[1]
	# row_b = np.shape(b)[0]
	# col_b = np.shape(b)[1]
	# return b_shape
	if a_shape[1] != b_shape[0] : return -1
	else:
		res = a.dot(b)
		return res