# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 12:40:35 2022

@author: hofer
"""

import numpy as np
from scipy.linalg import svd, cholesky, solve_triangular, diagsvd
from sklearn.datasets import make_spd_matrix
from numpy.linalg import inv, solve

length = 3
shape = (length, length)
a = make_spd_matrix(length)
b = np.ones(length)

print(a)
u, s, vt = svd(a)
S = diagsvd(s, *shape)
solve_svd = vt.T @ inv(S) @ u.T @ b

print(solve_svd)

sol = solve(a, b)
print(sol)

transform = cholesky(a)
csol = solve_triangular(transform, b)
print(csol)



