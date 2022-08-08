# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 12:05:22 2022

@author: hofer
"""
import numpy as np
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit
import time

EPS = np.finfo(float).eps


class CommonJIT():
    
    def __init__(self):
        # self.quadratic1, self.quadratic2 = self.create_quadratic_funcs()
        self.create_quadratic_funcs()
        self.create_js_dot()
        self.create_jac_sum()
        self.create_scale_for_robust_loss_function()
        
    
    def create_scale_for_robust_loss_function(self):

        @jit
        def scale_for_robust_loss_function(J, f, rho):
            """Scale Jacobian and residuals for a robust loss function.
            Arrays are modified in place.
            """
            J_scale = rho[1] + 2 * rho[2] * f**2
            mask = J_scale < EPS
            J_scale = jnp.where(mask, EPS, J_scale)
            J_scale = J_scale**0.5
            fscale = (rho[1] / J_scale)

            f = f * fscale
            J = J * J_scale[:, jnp.newaxis]
            return J, f
        self.scale_for_robust_loss_function = scale_for_robust_loss_function    
    
    
    def build_quadratic_1d(self, J, g, s, diag=None, s0=None):
        s_jnp = jnp.array(s)
        v_jnp = self.js_dot(J, s_jnp)
        v = v_jnp.copy()

        a = np.dot(v, v)
        if diag is not None:
            a += np.dot(s * diag, s)
        a *= 0.5
    
        b = np.dot(g, s)
    
        if s0 is not None:
            s0_jnp = jnp.array(s)
            u_jnp = self.js_dot(J, s0_jnp)
            u = u_jnp.copy()
            u = J.dot(s0)
            b += np.dot(u, v)
            c = 0.5 * np.dot(u, u) + np.dot(g, s0)
            if diag is not None:
                b += np.dot(s0 * diag, s)
                c += 0.5 * np.dot(s0 * diag, s0)
            return a, b, c
        else:
            return a, b
    
    
    def compute_jac_scale(self, J, scale_inv_old=None):
        """Compute variables scale based on the Jacobian matrix."""

        scale_inv_jnp = self.jac_sum_func(J)
        scale_inv = np.array(scale_inv_jnp)
    
        if scale_inv_old is None:
            scale_inv[scale_inv == 0] = 1
        else:
            scale_inv = np.maximum(scale_inv, scale_inv_old)

        return 1 / scale_inv, scale_inv
    

    def create_js_dot(self):
        @jit
        def js_dot(J, s):
            return J.dot(s)
        self.js_dot =  js_dot


    def create_jac_sum(self):
      @jit
      def jac_sum_func(J):
          return jnp.sum(J**2, axis=0)**0.5
      self.jac_sum_func = jac_sum_func
  
    
  
    def evaluate_quadratic(self, J, g, s_np, diag=None):
        s = jnp.array(s_np) #comes in as np array
        
        if s.ndim == 1:
            if diag is None:
                return self.evaluate_quadratic1(J, g, s)
            else:
                return self.evaluate_quadratic_diagonal1(J, g, s, diag)
        else:
            if diag is None:
                return self.evaluate_quadratic2(J, g, s)
            else:
                return self.evaluate_quadratic_diagonal2(J, g, s, diag)
    
    
    def create_quadratic_funcs(self):
        
        @jit
        def evaluate_quadratic1(J, g, s):
            Js = J.dot(s)
            q = jnp.dot(Js, Js)
            l = jnp.dot(s, g)
            return 0.5 * q + l
        
        @jit
        def evaluate_quadratic_diagonal1(J, g, s, diag):
            Js = J.dot(s)
            q = jnp.dot(Js, Js) + jnp.dot(s * diag, s)
            l = jnp.dot(s, g)
            return 0.5 * q + l
        
        @jit
        def evaluate_quadratic2(J, g, s):
            Js = J.dot(s.T)
            q = jnp.sum(Js**2, axis=0)
            l = jnp.dot(s, g)
            return 0.5 * q + l
        
        @jit
        def evaluate_quadratic_diagonal2(J, g, s, diag):
            Js = J.dot(s.T)
            q = jnp.sum(Js**2, axis=0) + jnp.sum(diag * s**2, axis=1)
            l = jnp.dot(s, g)
            return 0.5 * q + l
        
        self.evaluate_quadratic1 = evaluate_quadratic1
        self.evaluate_quadratic_diagonal1 = evaluate_quadratic_diagonal1
        self.evaluate_quadratic2 = evaluate_quadratic2
        self.evaluate_quadratic_diagonal2 = evaluate_quadratic_diagonal2
        
        
        
    # def evaluate_quadratic(self, J, g, s, diag=None):
    #     s_jnp = jnp.array(s)
    #     if s.ndim == 1:
    #         q_jnp =  self.quadratic1(J, s_jnp)
    #         q = q_jnp.copy()

    #         if diag is not None:
    #             q = q + np.dot(s * diag, s)
    #     else:
    #         # print('q2')
    #         q_jnp = self.quadratic2(J, s_jnp)
    #         q = q_jnp.copy()
    #         if diag is not None:
    #             q = q + np.sum(diag * s**2, axis=1)
        
    #     l = np.dot(s, g)
            
    #     return 0.5 * q + l


    # def create_quadratic_funcs(self):
    #     @jit
    #     def quadratic1(J, s):
    #         Js = J.dot(s)
    #         return jnp.dot(Js, Js)
    
    #     @jit
    #     def quadratic2(J, s):
    #         Js = J.dot(s.T)
    #         return jnp.sum(Js**2, axis=0)
    #     return quadratic1, quadratic2