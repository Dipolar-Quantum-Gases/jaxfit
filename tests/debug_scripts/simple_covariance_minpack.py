# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 19:05:04 2022

@author: hofer
"""
import numpy as np
import jax.numpy as jnp
import sys
sys.path.append('../')

from JAXFit import jax_curve_fit

from numpy.testing import (assert_, assert_almost_equal, assert_array_equal,
                           assert_array_almost_equal, assert_allclose,
                           assert_warns, suppress_warnings)

def func(x, a, b):
    return a * jnp.exp(-b*x)

def jac(x, a, b):
    e = jnp.exp(-b*x)
    return jnp.vstack((e, -a * x * e))

for flength in [None, 100]:
    curve_fit = jax_curve_fit(flength=flength).curve_fit
    
    np.random.seed(0)
    xdata = np.linspace(0, 4, 50)
    y = func(xdata, 2.5, 1.3)
    ydata = y + 0.2 * np.random.normal(size=len(xdata))

    sigma = np.zeros(len(xdata)) + 0.2
    covar = np.diag(sigma**2)
    
    if flength is not None:
        j = jac(xdata, 1, 1)
        print(type(j), j.shape)    
        print(j)

    # for jac1, jac2 in [(jac, jac), (None, None)]:

    # # for jac1, jac2 in [(None, None)]:

    #     for absolute_sigma in [False, True]:
    #         print(jac1, absolute_sigma)
    #         popt1, pcov1 = curve_fit(func, xdata, ydata, sigma=sigma,
    #                 jac=jac1, absolute_sigma=absolute_sigma)
    #         popt2, pcov2 = curve_fit(func, xdata, ydata, sigma=covar,
    #                 jac=jac2, absolute_sigma=absolute_sigma)

    #         assert_allclose(popt1, popt2, atol=1e-14)
    #         assert_allclose(pcov1, pcov2, atol=1e-14)