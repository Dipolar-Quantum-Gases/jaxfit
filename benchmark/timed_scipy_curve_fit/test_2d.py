# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 15:16:42 2022

@author: hofer
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from minpack import curve_fit



def get_coordinates(width, height):
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    X, Y = np.meshgrid(x, y)
    return X, Y
    

def rotate_coordinates2D(coordinates, theta):
    X, Y = coordinates
    shape = X.shape
    Xr, Yr = np.copy(X), np.copy(Y)
    coords = np.stack([Xr.flatten(), Yr.flatten()])
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    Xr, Yr = R @ coords
    return np.reshape(Xr, shape), np.reshape(Yr, shape)


def translate_coordinates2D(XY_tuple, x0, y0):
    X, Y = XY_tuple
    if x0 != 0:
        X = X - x0
    if y0 != 0:
        Y = Y - y0
    return (X, Y)
    

def coordinate_transformation2D(XY_tuple, x0=0, y0=0, theta=0):
    XY_tuple = translate_coordinates2D(XY_tuple, x0, y0)
    if theta != 0:
        XY_tuple = rotate_coordinates2D(XY_tuple, theta)
    return XY_tuple


def gaussian2d(XY_tuple, n0, x0, y0, sigma_x, sigma_y, theta, offset):
    X, Y = coordinate_transformation2D(XY_tuple, x0, y0, theta)
    gaussian_density = n0 * np.exp(-.5 * (X**2 / sigma_x**2 + Y**2 / sigma_y**2))
    return gaussian_density + offset


def get_random_float(low, high):
    delta = high - low
    return low + delta * np.random.random()


length = 400
XY_tuple = get_coordinates(length, length)


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
bounds = (min_bounds, max_bounds)

zdata = gaussian2d(XY_tuple, *params)

flat_XY = [coord.flatten() for coord in XY_tuple]
flat_data = zdata.flatten()

loop = 5
times = []
for i in range(loop):
    print(i)
    # seed = [param * get_random_float(.7, 1.4) for param in params]
    seed = [param * 1.3 for param in params]

    st = time.time()
    popt, pcov = curve_fit(gaussian2d, flat_XY, flat_data, p0=seed, 
                           method='trf', bounds=bounds)
    times.append(time.time() - st)
print(np.mean(times))

print(params)
print(popt)