# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 15:16:42 2022

@author: hofer
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys

sys.path.append(r'../')

from jaxfit import CurveFit
import jax.numpy as jnp

sys.path.append(r'./timed_scipy_curve_fit')
from scipy_minpack import curve_fit

sys.path.append('./classes')

from gauss2d_numpy import gaussian2d as gaussian2d_np
from gauss2d_jax import gaussian2d as gaussian2d_jax

def get_coordinates(width, height):
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    X, Y = np.meshgrid(x, y)
    return X, Y

def approximate_jonquieress_function(z, gamma, n_max=10):
    Li = jnp.zeros(z.shape)
    for n in range(1, n_max + 1):
        Li += z ** n / n ** gamma
    return Li
    

def rotate_coordinates2D(coords, theta):
    R = jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
                  [jnp.sin(theta), jnp.cos(theta)]])
    rcoords = R @ coords
    return rcoords


def translate_coordinates2D(coords, x0, y0):
    xcoords = coords[0] - x0  
    ycoords = coords[1] - y0
    return jnp.stack([xcoords, ycoords])
    

def coordinate_transformation2D(XY_tuple, x0=0, y0=0, theta=0):
    XY_tuple = translate_coordinates2D(XY_tuple, x0, y0)
    XY_tuple = rotate_coordinates2D(XY_tuple, theta)
    return XY_tuple
    

def parabola_base(coords, n0, Rx, Ry):
    X, Y = coords
    parabola = 1 - X**2 / Rx**2 - Y**2 / Ry**2
    parabola = parabola * (parabola > 0)
    return n0 * parabola


def thomas_fermi_bec(coords, n0, x0, y0, Rx, Ry, theta, offset):
    coords = coordinate_transformation2D(coords, x0, y0, theta)
    parabola = parabola_base(coords, 1, Rx, Ry)
    density = n0 * parabola**(3 / 2)
    return density + offset


def parabola2d(coords, n0, x0, y0, Rx, Ry, theta, offset):
    coords = coordinate_transformation2D(coords, x0, y0, theta)
    density = parabola_base(coords, n0, Rx, Ry)
    return density + offset


def gaussian2d_base(coords, n0, sigma_x, sigma_y):
    X, Y = coords
    return n0 * jnp.exp(-.5 * (X**2 / sigma_x**2 + Y**2 / sigma_y**2))


def gaussian2d(coords, n0, x0, y0, sigma_x, sigma_y, theta, offset):
    coords = coordinate_transformation2D(coords, x0, y0, theta)
    density = gaussian2d_base(coords, n0, sigma_x, sigma_y)
    return density + offset


def thermal_cloud(coords, n0, x0, y0, sigma_x, sigma_y, theta, offset):
    coords = coordinate_transformation2D(coords, x0, y0, theta)
    inside = gaussian2d_base(coords, 1, sigma_x, sigma_y)
    density = n0 * approximate_jonquieress_function(inside, 2)
    return density + offset

def wrap_1d_func(func, coords, function_parameters):
    if type(coords) is not np.array:
        coords = np.asarray(coords)
    if coords.ndim > 1:
        shape = coords[0].shape
        coords = [coord.flatten() for coord in coords]
    else:
        shape = coords.shape
    func_vals = func(coords, *function_parameters)
    return np.reshape(func_vals, shape)



def get_random_float(low, high):
    delta = high - low
    return low + delta * np.random.random()



length = 700
XY_tuple = get_coordinates(length, length)
jcf = CurveFit(flength=length**2)

n0 = 1
x0 = length / 2
y0 = length / 2
sigx = length / 6
sigy = length / 8
theta = np.pi / 3

offset = .1 * n0
params = [n0, x0, y0, sigx, sigy, theta, offset]
min_bounds = [-np.inf, -np.inf, -np.inf, 0, 0, -np.inf, -np.inf]
max_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
min_bounds = [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]
max_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
bounds = (min_bounds, max_bounds)

zdata = wrap_1d_func(gaussian2d, XY_tuple, params)
# zdata = gaussian2d(XY_tuple, *params)

flat_XY = [coord.flatten() for coord in XY_tuple]
# print(flat_XY.shape)
flat_data = zdata.flatten()
sigma = np.random.random(len(flat_data))
sigma = None

loop = 10
times = []
stimes = []

for i in range(loop):
    print(i)
    # seed = [param * get_random_float(.7, 1.4) for param in params]
    seed = [val * get_random_float(.7, 1.3) for val in params]


    st = time.time()
    popt, pcov = jcf.curve_fit(gaussian2d_jax, flat_XY, flat_data, p0=seed, 
                            method='trf', bounds=bounds, sigma=sigma)
                            # x_scale='jac', timeit=True)
    st2 = time.time()
    # popt, pcov = curve_fit(gaussian2d, flat_XY, flat_data, p0=seed, x_scale='jac')
    popt2, pcov2, res2, post_time2, ctime2 = curve_fit(gaussian2d_np, flat_XY, flat_data, p0=seed, 
                            method='trf', bounds=bounds, sigma=sigma)
                            # x_scale='jac')
    stimes.append(time.time() - st2)
    times.append(st2 - st)

    # res, post_time, ctime
    # stimes.append(jcf.res.all_times)
print(np.mean(times))

print(params)
print(popt)

plt.figure()
plt.plot(times[1:], label='JAXFit')
plt.plot(stimes[1:], label='SciPy')
plt.legend()
plt.show()

plt.figure()
plt.plot(np.array(stimes[1:]) / np.array(times[1:]))

plt.legend()
plt.show()