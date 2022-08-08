# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 18:07:04 2022

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

# Test that Jacobian callable is handled correctly and
# weighted if sigma is provided.
def f(x, a, b):
    return a * jnp.exp(-b*x)

def jac(x, a, b):
    e = jnp.exp(-b*x)
    return jnp.vstack((e, -a * x * e))

xdata = np.linspace(0, 1, 11)
ydata = np.array(f(xdata, 2., 2.))

for flength in [None, 100]:
    print('flength', flength)
    curve_fit = jax_curve_fit(flength=flength).curve_fit
    # Test numerical options for least_squares backend.
    for method in ['trf']:
        # for scheme in ['None', '2-point', '3-point', 'cs']:
        # for scheme in ['None']:
            popt, pcov = curve_fit(f, xdata, ydata, jac=None,
                                    method=method)
            print('popt', popt)
            assert_allclose(popt, [2, 2])
            
            
j = jac(xdata, 1, 1)
print(type(j), j.shape)

print(j)