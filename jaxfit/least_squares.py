"""Generic interface for least-squares minimization."""
from warnings import warn
import numpy as np
import time
from typing import Callable, Optional, Tuple, Union, Sequence, List, Any

from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, jacfwd
from jax.scipy.linalg import solve_triangular as jax_solve_triangular

from jaxfit.trf import TrustRegionReflective
from jaxfit.loss_functions import LossFunctionsJIT
from jaxfit.common_scipy import EPS, in_bounds, make_strictly_feasible


TERMINATION_MESSAGES = {
    -1: "Improper input parameters status returned from `leastsq`",
    0: "The maximum number of function evaluations is exceeded.",
    1: "`gtol` termination condition is satisfied.",
    2: "`ftol` termination condition is satisfied.",
    3: "`xtol` termination condition is satisfied.",
    4: "Both `ftol` and `xtol` termination conditions are satisfied."
}

def prepare_bounds(bounds, n) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare bounds for optimization.

    This function prepares the bounds for the optimization by ensuring that 
    they are both 1-D arrays of length `n`. If either bound is a scalar, it is 
    resized to an array of length `n`.

    Parameters
    ----------
    bounds : Tuple[np.ndarray, np.ndarray]
        The lower and upper bounds for the optimization.
    n : int
        The length of the bounds arrays.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The prepared lower and upper bounds arrays.
    """
    lb, ub = [np.asarray(b, dtype=float) for b in bounds]
    if lb.ndim == 0:
        lb = np.resize(lb, n)

    if ub.ndim == 0:
        ub = np.resize(ub, n)

    return lb, ub


def check_tolerance(ftol: float, 
                    xtol: float, 
                    gtol: float, 
                    method: str
                    ) -> Tuple[float, float, float]:
    """Check and prepare tolerance values for optimization.

    This function checks the tolerance values for the optimization and 
    prepares them for use. If any of the tolerances is `None`, it is set to 
    0. If any of the tolerances is lower than the machine epsilon, a warning 
    is issued and the tolerance is set to the machine epsilon. If all 
    tolerances are lower than the machine epsilon, a `ValueError` is raised.

    Parameters
    ----------
    ftol : float
        The tolerance for the optimization function value.
    xtol : float
        The tolerance for the optimization variable values.
    gtol : float
        The tolerance for the optimization gradient values.
    method : str
        The name of the optimization method.

    Returns
    -------
    Tuple[float, float, float]
        The prepared tolerance values.
    """
    def check(tol: float, name: str) -> float:
        if tol is None:
            tol = 0
        elif tol < EPS:
            warn("Setting `{}` below the machine epsilon ({:.2e}) effectively "
                 "disables the corresponding termination condition."
                 .format(name, EPS))
        return tol

    ftol = check(ftol, "ftol")
    xtol = check(xtol, "xtol")
    gtol = check(gtol, "gtol")

    if ftol < EPS and xtol < EPS and gtol < EPS:
        raise ValueError("At least one of the tolerances must be higher than "
                         "machine epsilon ({:.2e}).".format(EPS))

    return ftol, xtol, gtol


def check_x_scale(x_scale: Union[str, Sequence[float]],
                  x0: Sequence[float]
                  ) -> Union[str, Sequence[float]]:
    """Check and prepare the `x_scale` parameter for optimization.

    This function checks and prepares the `x_scale` parameter for the 
    optimization. `x_scale` can either be 'jac' or an array_like with positive
    numbers. If it's 'jac' the jacobian is used as the scaling. 

    Parameters
    ----------
    x_scale : Union[str, Sequence[float]]
        The scaling for the optimization variables.
    x0 : Sequence[float]
        The initial guess for the optimization variables.

    Returns
    -------
    Union[str, Sequence[float]]
        The prepared `x_scale` parameter.
    """

    if isinstance(x_scale, str) and x_scale == 'jac':
        return x_scale

    try:
        x_scale = np.asarray(x_scale, dtype=float)
        valid = np.all(np.isfinite(x_scale)) and np.all(x_scale > 0)
    except (ValueError, TypeError):
        valid = False

    if not valid:
        raise ValueError("`x_scale` must be 'jac' or array_like with "
                         "positive numbers.")

    if x_scale.ndim == 0:
        x_scale = np.resize(x_scale, x0.shape)

    if x_scale.shape != x0.shape:
        raise ValueError("Inconsistent shapes between `x_scale` and `x0`.")

    return x_scale

"""Wraps the given function such that a masked jacfwd is performed on it
thereby giving the autodiff jacobian."""
class AutoDiffJacobian():
    """Wraps the residual fit function such that a masked jacfwd is performed 
    on it. thereby giving the autodiff Jacobian. This needs to be a class since 
    we need to maintain in memory three different versions of the Jacobian.
    """

    def create_ad_jacobian(self, 
                           func: Callable, 
                           num_args: int, 
                           masked: bool = True
                           ) -> Callable:
        """Creates a function that returns the autodiff jacobian of the 
        residual fit function. The Jacobian of the residual fit function is
        equivalent to the Jacobian of the fit function.

        Parameters
        ----------
        func : Callable
            The function to take the jacobian of.
        num_args : int
            The number of arguments the function takes.
        masked : bool, optional
            Whether to use a masked jacobian, by default True

        Returns
        -------
        Callable
            The function that returns the autodiff jacobian of the given
            function.
        """

        # create a list of argument indices for the wrapped function which
        # will correspond to the arguments of the residual fit function and
        # will be past to JAX's jacfwd function.
        arg_list = [4 + i for i in range(num_args)]
            
        @jit
        def wrap_func(*all_args: List[Any]) -> jnp.ndarray:
            """Wraps the residual fit function such that it can be passed to the
            jacfwd function. Jacfwd requires the function to a single list
            of arguments.
            """
            xdata, ydata, data_mask, atransform = all_args[:4]
            args = all_args[4:]
            return func(args, xdata, ydata, data_mask, atransform)
        
        @jit
        def jac_func(args: List[float], 
                     xdata: jnp.ndarray, 
                     ydata: jnp.ndarray, 
                     data_mask: jnp.ndarray, 
                     atransform: jnp.ndarray
                     ) -> jnp.ndarray:
            """Returns the jacobian. Places all the residual fit function
            arguments into a single list for the wrapped residual fit function.
            Then calls the jacfwd function on the wrapped function with the
            the arglist of the arguments to differentiate with respect to which
            is only the arguments of the original fit function.
            """

            fixed_args = [xdata, ydata, data_mask, atransform] 
            all_args = [*fixed_args, *args]
            jac_fwd = jacfwd(wrap_func, argnums=arg_list)(*all_args)
            return jnp.array(jac_fwd)
        
        @jit
        def masked_jac(args: List[float], 
                       xdata: jnp.ndarray, 
                       ydata: jnp.ndarray, 
                       data_mask: jnp.ndarray, 
                       atransform: jnp.ndarray
                       ) -> jnp.ndarray:
            """Returns the masked jacobian."""
            Jt = jac_func(args, xdata, ydata, data_mask, atransform)
            J = jnp.where(data_mask, Jt, 0).T 
            return jnp.atleast_2d(J)
         
        @jit
        def no_mask_jac(args: List[float],
                        xdata: jnp.ndarray, 
                        ydata: jnp.ndarray,
                        data_mask: jnp.ndarray, 
                        atransform: jnp.ndarray
                        ) -> jnp.ndarray:
            """Returns the unmasked jacobian."""
            J = jac_func(args, xdata, ydata, data_mask, atransform).T
            return jnp.atleast_2d(J)
        
        if masked:
            self.jac = masked_jac
        else:
            self.jac = no_mask_jac
        return self.jac
    


class LeastSquares():
    
    def __init__(self):
        super().__init__() # not sure if this is needed
        self.trf = TrustRegionReflective()
        self.ls = LossFunctionsJIT()
        #initialize jacobian to None and f to a dummy function
        self.f = lambda x: None 
        self.jac = None
        
        # need a separate instance of the autodiff class for each of the 
        # the different sigma/covariance cases
        self.adjn = AutoDiffJacobian()
        self.adj1d = AutoDiffJacobian()
        self.adj2d = AutoDiffJacobian()
    

    def least_squares(self, 
                      fun: Callable, 
                      x0: np.ndarray, 
                      jac: Optional[Callable] = None, 
                      bounds: Tuple[np.ndarray, np.ndarray] = (-np.inf, np.inf),
                      method: str = 'trf',
                      ftol: float = 1e-8,
                      xtol: float = 1e-8, 
                      gtol: float = 1e-8, 
                      x_scale: Union[str, np.ndarray, float] = 1.0,
                      loss: str = 'linear',
                      f_scale: float = 1.0,
                      diff_step=None, 
                      tr_solver=None, 
                      tr_options={},
                      jac_sparsity=None, 
                      max_nfev: Optional[float] = None, 
                      verbose: int = 0, 
                      xdata: Optional[jnp.ndarray] = None, 
                      ydata: Optional[jnp.ndarray] = None, 
                      data_mask: Optional[jnp.ndarray] = None, 
                      transform: Optional[jnp.ndarray] = None, 
                      timeit: bool = False,
                      args=(), 
                      kwargs={}):
        
        
        if data_mask is None and ydata is not None:
            data_mask = jnp.ones(len(ydata), dtype=bool)
            
        if loss not in self.ls.IMPLEMENTED_LOSSES and not callable(loss):
            raise ValueError("`loss` must be one of {0} or a callable."
                             .format(self.ls.IMPLEMENTED_LOSSES.keys()))
        
        if method not in ['trf']:
            raise ValueError("`method` must be 'trf")
            
        if jac not in [None] and not callable(jac):
            raise ValueError("`jac` must be None or "
                             "callable.")
    
        if verbose not in [0, 1, 2]:
            raise ValueError("`verbose` must be in [0, 1, 2].")
    
        if len(bounds) != 2:
            raise ValueError("`bounds` must contain 2 elements.")
    
        if max_nfev is not None and max_nfev <= 0:
            raise ValueError("`max_nfev` must be None or positive integer.")
    
        if np.iscomplexobj(x0):
            raise ValueError("`x0` must be real.")
    
        x0 = np.atleast_1d(x0).astype(float)
    
        if x0.ndim > 1:
            raise ValueError("`x0` must have at most 1 dimension.")
            
        self.n = len(x0)

    
        lb, ub = prepare_bounds(bounds, x0.shape[0])
    
        if lb.shape != x0.shape or ub.shape != x0.shape:
            raise ValueError("Inconsistent shapes between bounds and `x0`.")
    
        if np.any(lb >= ub):
            raise ValueError("Each lower bound must be strictly less than each "
                             "upper bound.")
    
        if not in_bounds(x0, lb, ub):
            raise ValueError("`x0` is infeasible.")
            
    
        x_scale = check_x_scale(x_scale, x0)
        ftol, xtol, gtol = check_tolerance(ftol, xtol, gtol, method)
        x0 = make_strictly_feasible(x0, lb, ub)
        

        if xdata is not None and ydata is not None:
            # checks to see if the fit function is the same. Can't directly
            # compare the functions so we compare function code directly
            func_update = self.f.__code__.co_code != fun.__code__.co_code
            # if we are updating the fit function then we need to update the
            # jacobian function as well
            if func_update:
                self.update_function(fun)
                # this only updates the the jacobian if using autodiff (jac=None)
                if jac is None:
                    self.autdiff_jac(jac)
                    
            # if using an analytical jacobian
            if jac is not None:
                # if we are in the first function call
                if self.jac is None:
                    self.wrap_jac(jac)
                elif self.jac.__code__.co_code != jac.__code__.co_code:
                    # checks to see if the jacobian function is the same (see
                    # func_update for why no direct comparing of the functions)
                    # if it's a different Jacobian we need to rewrap it
                    self.wrap_jac(jac)
            elif self.jac is not None and not func_update:
                self.autdiff_jac(jac)
        
            # determines the correct residual function and jacobian to use
            # depending on whether data uncertainty transform is None, 1D, or 2D
            if transform is None:
                rfunc = self.func_none
                jac_func = self.jac_none
            elif transform.ndim == 1:
                rfunc = self.func_1d
                jac_func = self.jac_1d
            else:
                rfunc = self.func_2d
                jac_func = self.jac_2d
        else:
            # this if/else is to maintain compatibility with the SciPy suite of tests
            # which assume the residual function contains the fit data which is not
            # the case for JAXFit due to how we've made the residual function
            # function to be compatible with JAX JIT compilation
            def wrap_func(fargs, xdata, ydata, data_mask, atransform):
                return jnp.atleast_1d(fun(fargs, *args, **kwargs))
            
            def wrap_jac(fargs, xdata, ydata, data_mask, atransform):
                return jnp.atleast_2d(jac(fargs, *args, **kwargs))
            
            rfunc = wrap_func
            if jac is None:
                adj = AutoDiffJacobian()
                jac_func = adj.create_ad_jacobian(wrap_func, self.n, masked=False)
            else:
                jac_func = wrap_jac
            
                
                
        f0 = rfunc(x0, xdata, ydata, data_mask, transform)
        J0 = jac_func(x0, xdata, ydata, data_mask, transform)

        if f0.ndim != 1:
            raise ValueError("`fun` must return at most 1-d array_like. "
                             "f0.shape: {0}".format(f0.shape))

        if not np.all(np.isfinite(f0)):
            raise ValueError("Residuals are not finite in the initial point.")

        n = x0.size
        m = f0.size
        
        if J0 is not None:
            if J0.shape != (m, n):
                raise ValueError(
                    "The return value of `jac` has wrong shape: expected {0}, "
                    "actual {1}.".format((m, n), J0.shape))
                
        if data_mask is None:
            data_mask = jnp.ones(m)
        
        loss_function = self.ls.get_loss_function(loss)

        if callable(loss):
            rho = loss_function(f0, f_scale, data_mask=data_mask)
            if rho.shape != (3, m):
                raise ValueError("The return value of `loss` callable has wrong "
                                 "shape.")
            initial_cost_jnp = self.trf.calculate_cost(rho, data_mask)
        elif loss_function is not None:
            initial_cost_jnp = loss_function(f0, f_scale, data_mask=data_mask, 
                                             cost_only=True)
        else:
            initial_cost_jnp = self.trf.default_loss_func(f0)
        initial_cost = np.array(initial_cost_jnp)
        
        result = self.trf.trf(rfunc, xdata, ydata, jac_func, data_mask, 
                              transform, x0, f0, J0, lb, ub, ftol, xtol,
                     gtol, max_nfev, f_scale, x_scale, loss_function,
                     tr_options.copy(), verbose, timeit)
    
    
        result.message = TERMINATION_MESSAGES[result.status]
        result.success = result.status > 0
    
        if verbose >= 1:
            print(result.message)
            print("Function evaluations {0}, initial cost {1:.4e}, final cost "
                  "{2:.4e}, first-order optimality {3:.2e}."
                  .format(result.nfev, initial_cost, result.cost,
                          result.optimality))
            
        return result
   
            
    def autdiff_jac(self, jac: None) -> None:
        """We do this for all three sigma transformed functions such
        that if sigma is changed from none to 1D to covariance sigma then no
        retracing is needed.

        Parameters
        ----------
        jac : None
            Passed in to maintain compatibility with the user defined Jacobian
            function.
        """
        self.jac_none = self.adjn.create_ad_jacobian(self.func_none, self.n)
        self.jac_1d = self.adj1d.create_ad_jacobian(self.func_1d, self.n)
        self.jac_2d = self.adj2d.create_ad_jacobian(self.func_2d, self.n)
        # jac is
        self.jac = jac


    def update_function(self, func: Callable) -> None:
        """Wraps the given fit function to be a residual function using the
        data. The wrapped function is in a JAX JIT compatible format which
        is purely functional. This requires that both the data mask and the
        uncertainty transform are passed to the function. Even for the case
        where the data mask is all True and the uncertainty transform is None
        we still need to pass these arguments to the function due JAX's
        functional nature.

        Parameters
        ----------
        func : Callable
            The fit function to wrap.

        Returns
        -------
        None
        """
 
        @jit
        def masked_residual_func(args: List[float], 
                                 xdata: jnp.ndarray, 
                                 ydata: jnp.ndarray, 
                                 data_mask: jnp.ndarray
                                 ) -> jnp.ndarray:

            """Compute the residual of the function evaluated at `args` with 
            respect to the data.

            This function computes the residual of the user fit function 
            evaluated at `args` with respect to the data `(xdata, ydata)`, 
            masked by `data_mask`. The residual is defined as the difference 
            between the function evaluation and the data. The masked residual 
            is obtained by setting the residual to 0 wherever the corresponding 
            element of `data_mask` is 0.

            Parameters
            ----------
            args : List[float]
                The parameters of the function.
            xdata : jnp.ndarray
                The independent variable data.
            ydata : jnp.ndarray
                The dependent variable data.
            data_mask : jnp.ndarray
                The mask for the data.

            Returns
            -------
            jnp.ndarray
                The masked residual of the function evaluated at `args` with respect to the data.
            """
            func_eval = func(xdata, *args) - ydata
            return jnp.where(data_mask, func_eval, 0)

        # need to define a separate function for each of the different
        # sigma/covariance cases as the uncertainty transform is different
        # for each case. In future could remove the no transfore bit by setting
        # the uncertainty transform to all ones in the case where there is no
        # uncertainty transform.
        
        @jit
        def func_no_transform(args: List[float], 
                              xdata: jnp.ndarray, 
                              ydata: jnp.ndarray, 
                              data_mask: jnp.ndarray, 
                              atransform: jnp.ndarray
                              ) -> jnp.ndarray:

            """The residual function when there is no uncertainty transform.
            The atranform argument is not used in this case, but is included
            for consistency with the other cases."""
            return masked_residual_func(args, xdata, ydata, data_mask)
        
        @jit
        def func_1d_transform(args: List[float], 
                              xdata: jnp.ndarray, 
                              ydata: jnp.ndarray, 
                              data_mask: jnp.ndarray, 
                              atransform: jnp.ndarray
                              ) -> jnp.ndarray:
            """The residual function when there is a 1D uncertainty transform,
            that is when only the diagonal elements of the inverse covariance
            matrix are used."""
            return atransform * masked_residual_func(args, xdata, 
                                                     ydata, data_mask)
        
        @jit
        def func_2d_transform(args: List[float], 
                              xdata: jnp.ndarray, 
                              ydata: jnp.ndarray, 
                              data_mask: jnp.ndarray, 
                              atransform: jnp.ndarray
                              ) -> jnp.ndarray:
            """The residual function when there is a 2D uncertainty transform,
            that is when the full covariance matrix is given."""
            f = masked_residual_func(args, xdata, ydata, data_mask)
            return jax_solve_triangular(atransform, f, lower=True)
        
        self.func_none = func_no_transform
        self.func_1d = func_1d_transform
        self.func_2d = func_2d_transform
        self.f = func
    

    def wrap_jac(self, jac: Callable) -> None:
        """Wraps an user defined Jacobian function to allow for data masking
        and uncertainty transforms. The wrapped function is in a JAX JIT
        compatible format which is purely functional. This requires that both
        the data mask and the uncertainty transform are passed to the function.
        
        Using an analytical Jacobian of the fit function is equivalent to
        the Jacobian of the residual function.

        Also note that the analytical Jacobian doesn't require the independent
        ydata, but we still need to pass it to the function to maintain
        compatibility with autdiff version which does require the ydata.
        
        Parameters
        ----------
        jac : Callable
            The Jacobian function to wrap.
            
        Returns
        -------
        jnp.ndarray
            The masked Jacobian of the function evaluated at `args` with respect to the data.
        """
        
        @jit
        def jac_func(coords: jnp.ndarray,
                     args: List[float]
                     ) -> jnp.ndarray:
            jac_fwd = jac(coords, *args)
            return jnp.array(jac_fwd)
        
        @jit
        def masked_jac(coords: jnp.ndarray, 
                       args: List[float], 
                       data_mask: jnp.ndarray
                       ) -> jnp.ndarray:
             """Compute the wrapped Jacobian but masks out the padded elements 
             with 0s"""
             Jt = jac_func(coords, args)
             return jnp.where(data_mask, Jt, 0).T        
        
        @jit
        def jac_no_transform(args: List[float], 
                             coords: jnp.ndarray, 
                             ydata: jnp.ndarray, 
                             data_mask: jnp.ndarray, 
                             atransform: jnp.ndarray
                             ) -> jnp.ndarray:
            """The wrapped Jacobian function when there is no 
            uncertainty transform."""
            return jnp.atleast_2d(masked_jac(coords, args, data_mask))

        @jit
        def jac_1d_transform(args: List[float], 
                             coords: jnp.ndarray, 
                             ydata: jnp.ndarray, 
                             data_mask: jnp.ndarray, 
                             atransform: jnp.ndarray
                             ) -> jnp.ndarray:
            """The wrapped Jacobian function when there is a 1D uncertainty
            transform, that is when only the diagonal elements of the inverse
            covariance matrix are used."""
            J = masked_jac(coords, args, data_mask)
            return jnp.atleast_2d(atransform[:, jnp.newaxis] * jnp.asarray(J))
        
        @jit
        def jac_2d_transform(args: List[float], 
                             coords: jnp.ndarray, 
                             ydata: jnp.ndarray, 
                             data_mask: jnp.ndarray, 
                             atransform: jnp.ndarray
                             ) -> jnp.ndarray:
            """The wrapped Jacobian function when there is a 2D uncertainty
            transform, that is when the full covariance matrix is given."""

            J = masked_jac(coords, args, data_mask)
            return jnp.atleast_2d(jax_solve_triangular(atransform, 
                                                       jnp.asarray(J), 
                                                       lower=True))
        # we need all three versions of the Jacobian function to allow for
        # changing the sigma transform from none to 1D to 2D without having
        # to retrace the function
        self.jac_none = jac_no_transform
        self.jac_1d = jac_1d_transform
        self.jac_2d = jac_2d_transform
        self.jac = jac