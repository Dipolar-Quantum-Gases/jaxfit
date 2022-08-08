# -*- coding: utf-8 -*-
"""
Created on Tue May 10 16:22:03 2022

@author: hofer
"""

from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp


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


def gaussian2d_base(coords, n0, sigma_x, sigma_y):
    X, Y = coords
    return n0 * jnp.exp(-.5 * (X**2 / sigma_x**2 + Y**2 / sigma_y**2))


def gaussian2d(coords, n0, x0, y0, sigma_x, sigma_y, theta, offset):
    coords = coordinate_transformation2D(coords, x0, y0, theta)
    density = gaussian2d_base(coords, n0, sigma_x, sigma_y)
    return density + offset