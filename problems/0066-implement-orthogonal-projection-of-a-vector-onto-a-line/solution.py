def dot(a,b):
	res = 0
	for i in range(len(a)):
		res += a[i]*b[i]
	
	return res

def orthogonal_projection(v, L):
	"""
	Compute the orthogonal projection of vector v onto line L.

	:param v: The vector to be projected
	:param L: The line vector defining the direction of projection
	:return: List representing the projection of v onto L
	"""
	nom = dot(v,L)
	den = dot(L,L)

	scalar = nom/den
	return [scalar*i for i in L]
