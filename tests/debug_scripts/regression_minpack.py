# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 14:57:00 2022

@author: hofer
"""

import matplotlib.pyplot as plt
import sys
sys.path.append(r'G:\My Drive\nn_research\gpu_curve_fit\python\scipy')
from scipy_minpack import curve_fit


import sys
sys.path.append(r'G:\My Drive\nn_research\gpu_curve_fit\python\JAXFit\JAXFit')

from JAXFit import jax_curve_fit
import jax.numpy as jnp
jcf = jax_curve_fit()
curve_fit2 = jcf.curve_fit


import jax.numpy as jnp
from numpy.testing import assert_allclose
import numpy as np
np.set_printoptions(precision=4)



x = [574.14200000000005, 574.154, 574.16499999999996,
     574.17700000000002, 574.18799999999999, 574.19899999999996,
     574.21100000000001, 574.22199999999998, 574.23400000000004,
     574.245]
y = [859.0, 997.0, 1699.0, 2604.0, 2013.0, 1964.0, 2435.0,
     1550.0, 949.0, 841.0]
guess = [574.1861428571428, 574.2155714285715, 1302.0, 1302.0,
         0.0035019999999983615, 859.0]
good = [5.74177150e+02, 5.74209188e+02, 1.74187044e+03, 1.58646166e+03,
        1.0068462e-02, 8.57450661e+02]

def f_double_gauss(x, x0, x1, A0, A1, sigma, c):
    return (A0*jnp.exp(-(x-x0)**2/(2.*sigma**2))
            + A1*jnp.exp(-(x-x1)**2/(2.*sigma**2)) + c)

popt1, pcov,_, _, _ = curve_fit(f_double_gauss, x, y, guess, maxfev=10000, verbose=2, method='trf')
assert_allclose(popt1, good, rtol=1e-5)

popt2, pcov = curve_fit2(f_double_gauss, x, y, guess, maxfev=10000, verbose=2, method='trf')
assert_allclose(popt2, good, rtol=1e-5)


xmin = np.amin(x)
xmax = np.amax(x)
xp = np.linspace(xmin, xmax, 1000)
yps = f_double_gauss(xp, *guess)
yp1 = f_double_gauss(xp, *popt1)

yp2 = f_double_gauss(xp, *popt2)
plt.plot(xp, yps, label = 'seed')
plt.plot(xp, yp1, label = 'scipy')
plt.plot(xp, yp2, label = 'jax')
plt.legend()
plt.show()