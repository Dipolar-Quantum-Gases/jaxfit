"""
These are functions that were initially in the common.py file, but are have
large data operations and are therefore better suited to be compiled with
JAX.  They are compiled with JAX and then added to the CommonJIT class.
"""
import numpy as np
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit
import time
from typing import Tuple, List, Dict, Union, Optional, Callable

EPS = np.finfo(float).eps


class CommonJIT():
    
    def __init__(self):
        """Initialize the class and create the JAX/JIT functions that will be 
        compiled"""
        self.create_quadratic_funcs()
        self.create_js_dot()
        self.create_jac_sum()
        self.create_scale_for_robust_loss_function()
        
    
    def create_scale_for_robust_loss_function(self):
        """Create the scaling function for the loss functions"""

        @jit
        def scale_for_robust_loss_function(J: jnp.ndarray, 
                                           f: jnp.ndarray,
                                           rho: jnp.ndarray
                                           ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """Scale Jacobian and residuals for a robust loss function.
            Arrays are modified in place.

            Parameters
            ----------
            J : jnp.ndarray
                Jacobian matrix.
            f : jnp.ndarray
                Residuals.
            rho : jnp.ndarray
                Cost function evaluation.
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
    
    
    def build_quadratic_1d(self, 
                           J: jnp.ndarray, 
                           g: jnp.ndarray, 
                           s: jnp.ndarray, 
                           diag: Optional[jnp.ndarray] = None, 
                           s0: Optional[jnp.ndarray] = None
                           ) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], 
                                      Tuple[np.ndarray, np.ndarray]]:
        """Parameterize a multivariate quadratic function along a line.
        The resulting univariate quadratic function is given as follows:
        ::
            f(t) = 0.5 * (s0 + s*t).T * (J.T*J + diag) * (s0 + s*t) + g.T * (s0 + s*t)

        Parameters
        ----------
        J : ndarray, sparse matrix or LinearOperator shape (m, n)
            Jacobian matrix, affects the quadratic term.
        g : ndarray, shape (n,)
            Gradient, defines the linear term.
        s : ndarray, shape (n,)
            Direction vector of a line.
        diag : None or ndarray with shape (n,), optional
            Addition diagonal part, affects the quadratic term.
            If None, assumed to be 0.
        s0 : None or ndarray with shape (n,), optional
            Initial point. If None, assumed to be 0.

        Returns
        -------
        a : float
            Coefficient for t**2.
        b : float
            Coefficient for t.
        c : float
            Free term. Returned only if `s0` is provided.

        """
        
        s_jnp = jnp.array(s)
        v_jnp = self.js_dot(J, s_jnp)
        v = v_jnp.copy()

        a = np.dot(v, v)
        if diag is not None:
            a += np.dot(s * diag, s)
        a *= 0.5
    
        b = np.dot(g, s)
    
        if s0 is not None:
            s0_jnp = jnp.array(s0)
            u_jnp = self.js0_dot(J, s0_jnp)
            u = u_jnp.copy()

            b += np.dot(u, v)
            c = 0.5 * np.dot(u, u) + np.dot(g, s0)
            if diag is not None:
                b += np.dot(s0 * diag, s)
                c += 0.5 * np.dot(s0 * diag, s0)
            return a, b, c
        else:
            return a, b


    def compute_jac_scale(self, J: jnp.ndarray, 
                          scale_inv_old: Optional[np.ndarray] = None
                          ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute variables scale based on the Jacobian matrix.
        
        Parameters
        ----------
        J : jnp.ndarray
            Jacobian matrix.
        scale_inv_old : Optional[np.ndarray], optional
            Previous scale, by default None

        Returns
        -------
        scale : np.ndarray
            Scale for the variables.
        scale_inv : np.ndarray
            Inverse of the scale for the variables.
        """

        scale_inv_jnp = self.jac_sum_func(J)
        scale_inv = np.array(scale_inv_jnp)
    
        if scale_inv_old is None:
            scale_inv[scale_inv == 0] = 1
        else:
            scale_inv = np.maximum(scale_inv, scale_inv_old)

        return 1 / scale_inv, scale_inv
    

    def create_js_dot(self):
        """Create the functions for the dot product of the Jacobian and the
        search direction. We need two functions because s and s0 are different
        shapes which causes retracing of the function if we use the same
        function for both.
        """
        @jit
        def js_dot(J: jnp.ndarray, s: jnp.ndarray) -> jnp.ndarray:
            return J.dot(s)

        @jit
        def js0_dot(J: jnp.ndarray, s0: jnp.ndarray) -> jnp.ndarray:
            return J.dot(s0)

        self.js_dot = js_dot
        self.js0_dot = js0_dot

  
    def evaluate_quadratic(self, 
                           J: jnp.ndarray, 
                           g: jnp.ndarray, 
                           s_np: np.ndarray, 
                           diag: Optional[np.ndarray] = None
                           ) -> jnp.ndarray:
        """Compute values of a quadratic function arising in least squares.
        The function is 0.5 * s.T * (J.T * J + diag) * s + g.T * s.

        Parameters
        ----------
        J : ndarray, sparse matrix or LinearOperator, shape (m, n)
            Jacobian matrix, affects the quadratic term.
        g : ndarray, shape (n,)
            Gradient, defines the linear term.
        s : ndarray, shape (k, n) or (n,)
            Array containing steps as rows.
        diag : ndarray, shape (n,), optional
            Addition diagonal part, affects the quadratic term.
            If None, assumed to be 0.
        Returns
        -------
        values : ndarray with shape (k,) or float
            Values of the function. If `s` was 2-D, then ndarray is
            returned, otherwise, float is returned.
        """
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


    def create_jac_sum(self):
        """Create the function for the sum of the Jacobian squared and then
        taking the square root. This is used to compute the scale for the
        variables. Can potentially remove this.
        """
        @jit
        def jac_sum_func(J):
            return jnp.sum(J**2, axis=0)**0.5
        self.jac_sum_func = jac_sum_func