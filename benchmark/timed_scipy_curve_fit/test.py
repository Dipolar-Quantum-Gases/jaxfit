# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 15:16:42 2022

@author: hofer
"""

import numpy as np
import matplotlib.pyplot as plt


from minpack import curve_fit



def func(x, a, b, c):
    return a * x**2 + b * x + c

a, b, c = 2, 3, 4
x = np.linspace(-5, 5, 100)
y = func(x, a, b, c)

plt.figure()
plt.plot(x, y)

popt, pcov =  curve_fit(func, x, y, method='trf')