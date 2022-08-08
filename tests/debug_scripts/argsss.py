# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 16:56:36 2022

@author: hofer
"""

import jax.numpy as jnp
from jax import jacfwd, jit



def func(arg1, arg2, *args):
    print(arg1)
    print(args)
    all_args = arg1 + list(args)
    return jnp.product(jnp.array(all_args))

arg1 = [1.0, 2.0, 3.0]
arg2 = [3.0, 4.0]
arg = [i for i in range(len(arg1))]


lfunc = lambda x : func(x, *arg2)
def jac_func(arg1, *args):
    all_args = arg1 + list(args)
    jac_fwd = jacfwd(func)(all_args)
    # jac_fwd = jacfwd(func, argnums=arg_list)(*args)
    return jnp.array(jac_fwd)

print(jac_func(arg1, *arg2))



