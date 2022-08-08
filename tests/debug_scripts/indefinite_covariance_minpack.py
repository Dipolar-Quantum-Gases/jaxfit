# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 16:01:21 2022

@author: hofer
"""
from numpy.testing import (assert_, assert_almost_equal, assert_array_equal,
                           assert_array_almost_equal, assert_allclose,
                           assert_warns, suppress_warnings)

from scipy.optimize import OptimizeWarning
import matplotlib.pyplot as plt
import sys
sys.path.append(r'G:\My Drive\nn_research\gpu_curve_fit\python\scipy')
from scipy_minpack import curve_fit
from scipy.optimize import curve_fit



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





xdata = np.array([1, 2, 3, 4, 5, 6])
ydata = np.array([1, 2, 3, 4, 5.5, 6])
assert_warns(OptimizeWarning, curve_fit,
             lambda x, a, b: a*x, xdata, ydata)


curve_fit(lambda x, a, b: a*x, xdata, ydata, method='trf')