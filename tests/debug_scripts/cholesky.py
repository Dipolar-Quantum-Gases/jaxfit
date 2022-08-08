# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 11:07:29 2022

@author: hofer
"""

import numpy as np
from scipy.linalg import svd, cholesky, solve_triangular, diagsvd
from sklearn.datasets import make_spd_matrix
from numpy.linalg import inv

# xdata = np.linspace(0, 3, 4)
# sigma = np.zeros(len(xdata)) + 0.2
# covar = np.diag(sigma**2)
# 
# 
# transform = cholesky(covar, lower=True)
# print(covar)
# print(transform)
# print('------')
# # %%
# np.random.seed(0)
# xdata = np.arange(1, 4)
# # y = func(xdata, 2.5, 1.0)
# # ydata = y + 0.2 * np.random.normal(size=len(xdata))
# sigma = np.zeros(len(xdata)) + 0.2
# covar = np.diag(sigma**2)
# transform = cholesky(covar, lower=True)

# print(covar)
# print(transform)
# print('------')

# %%
length = 3
shape = (length,length)
randint = np.random.randint(3, size=shape)
lt = np.tril(randint)
ut = lt.transpose()
covar = np.zeros(shape)
# covar = lt + ut
covar = make_spd_matrix(length)
print(covar)
transform = cholesky(covar, lower=True)
print(transform)

slicev = length - 1

covar_small = covar.copy()
covar_small = np.delete(covar_small, slicev, 1)
covar_small = np.delete(covar_small, slicev, 0)
print(covar_small)
transform_small = cholesky(covar_small, lower=True)
print(transform_small)

try:
    covar_rem = covar.copy()
    covar_rem[slicev] = np.zeros(length)
    covar_rem[:,slicev] = np.zeros(length)
    print(covar_rem)
    transform_rem = cholesky(covar_rem, lower=True)
    print(transform_rem)
except:
    print('couldnt do cholesky on padded covariance')

tshape = transform_small.shape
dshape = np.array(shape) - np.array(tshape)
transform_large = np.pad(transform_small, ((0,dshape[0]),(0, dshape[1])), 'constant')

b = np.ones(len(transform_small))
solve_small = solve_triangular(transform_small, b)
# solve_large = solve_triangular(transform_large, np.ones(len(transform_large)))

print(solve_small)
# print(solve_large)

u, s, vt = svd(covar_small)
S = diagsvd(s, *tshape)
Sinv = S.T

solve_svd = vt.T.dot(Sinv).dot(u.T).dot(b)
print(solve_svd)



