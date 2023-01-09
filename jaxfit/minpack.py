import warnings
import numpy as np
from numpy import (zeros, inf)
from inspect import signature
import time
from typing import Optional, Callable, Tuple, Union, List, Dict, Any

from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import jit
from jax.scipy.linalg import svd as jax_svd
from jax.scipy.linalg import cholesky as jax_cholesky

from jaxfit._optimize import OptimizeWarning
from jaxfit.least_squares import prepare_bounds, LeastSquares
from jaxfit.common_scipy import EPS

__all__ = ['CurveFit', 'curve_fit']

def curve_fit(f, *args, **kwargs):
    jcf = CurveFit(f)
    popt, pcov, _, _, _ = jcf.curve_fit(*args, **kwargs)
    return popt, pcov


def _initialize_feasible(lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Initialize feasible parameters for optimization.

    This function initializes feasible parameters for optimization based on the 
    lower and upper bounds of the variables. If both bounds are finite, the 
    feasible parameters are set to the midpoint between the bounds. If only the 
    lower bound is finite, the feasible parameters are set to the lower bound 
    plus 1. If only the upper bound is finite, the feasible parameters are set 
    to the upper bound minus 1. If neither bound is finite, the feasible 
    parameters are set to 1.

    Parameters
    ----------
    lb : np.ndarray
        The lower bounds of the variables.
    ub : np.ndarray
        The upper bounds of the variables.

    Returns
    -------
    np.ndarray
        The initialized feasible parameters.
    """

    p0 = np.ones_like(lb)
    lb_finite = np.isfinite(lb)
    ub_finite = np.isfinite(ub)

    mask = lb_finite & ub_finite
    p0[mask] = 0.5 * (lb[mask] + ub[mask])

    mask = lb_finite & ~ub_finite
    p0[mask] = lb[mask] + 1

    mask = ~lb_finite & ub_finite
    p0[mask] = ub[mask] - 1

    return p0


class CurveFit():
    
    def __init__(self, flength: Optional[float] = None):
        """CurveFit class for fitting

        Parameters
        ----------
        flength : float, optional
            fixed data length for fits, JAXFit pads input data to this length 
            to avoid retracing.
        """
        self.flength = flength
        self.create_sigma_transform_funcs()
        self.create_covariance_svd()
        
        self.ls = LeastSquares()
                
        
    def create_sigma_transform_funcs(self):
        """Create JIT-compiled sigma transform functions.

        This function creates two JIT-compiled functions: `sigma_transform1d` and 
        `sigma_transform2d`, which are used to compute the sigma transform for 1D 
        and 2D data, respectively. The functions are stored as attributes of the 
        object on which the method is called.
        """

        @jit
        def sigma_transform1d(sigma: jnp.ndarray, 
                      data_mask: jnp.ndarray
                     ) -> jnp.ndarray:
            """Compute the sigma transform for 1D data.

            Parameters
            ----------
            sigma : jnp.ndarray
                The standard deviation of the data.
            data_mask : jnp.ndarray
                A binary mask indicating which data points to use in the fit.

            Returns
            -------
            jnp.ndarray
                The sigma transform for the data.
            """
            transform = 1.0 / sigma
            return transform

        @jit
        def sigma_transform2d(sigma: jnp.ndarray, 
                              data_mask: jnp.ndarray
                              ) -> jnp.ndarray:
            """Compute the sigma transform for 2D data.

            Parameters
            ----------
            sigma : jnp.ndarray
                The standard deviation of the data.
            data_mask : jnp.ndarray
                A binary mask indicating which data points to use in the fit.

            Returns
            -------
            jnp.ndarray
                The sigma transform for the data.
            """
            sigma = jnp.asarray(sigma)
            transform = jax_cholesky(sigma, lower=True)
            return transform

        self.sigma_transform1d = sigma_transform1d
        self.sigma_transform2d = sigma_transform2d
        """For fixed input arrays we need to pad the actual data to match the 
        fixed input array size"""

    def create_covariance_svd(self): 
        """Create JIT-compiled SVD function for covariance computation."""
        @jit
        def covariance_svd(jac):
            _, s, VT = jax_svd(jac, full_matrices=False)
            return s, VT
        self.covariance_svd = covariance_svd
        
        
    def pad_fit_data(self, xdata: np.ndarray, 
                     ydata: np.ndarray, 
                     xdims: int, 
                     len_diff: int
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """Pad fit data to match the fixed input data length.

        This function pads the input data arrays with small values to match the 
        fixed input data length to avoid JAX retracing the JITted functions.
        The padding is added along the second dimension of the `xdata` array 
        if it's multidimensional data otherwise along the first dimension. The 
        small values are chosen to be `EPS`, a global constant defined as a 
        very small positive value which avoids numerical issues.

        Parameters
        ----------
        xdata : np.ndarray
            The independent variables of the data.
        ydata : np.ndarray
            The dependent variables of the data.
        xdims : int
            The number of dimensions in the `xdata` array.
        len_diff : int
            The difference in length between the data arrays and the fixed input data length.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The padded `xdata` and `ydata` arrays.
        """
        
        if xdims > 1:
            xpad = EPS * np.ones([xdims, len_diff])
            xdata = np.concatenate([xdata, xpad], axis=1)
        else:
            xpad = EPS * np.ones([len_diff])
            xdata = np.concatenate([xdata, xpad])
        ypad = EPS * np.ones([len_diff])
        ydata = np.concatenate([ydata, ypad])
        return xdata, ydata

        
        
    def curve_fit(self, 
                  f: Callable, 
                  xdata: Union[np.ndarray, Tuple[np.ndarray]],
                  ydata: np.ndarray, 
                  p0: Optional[np.ndarray] = None, 
                  sigma: Optional[np.ndarray] = None, 
                  absolute_sigma: bool = False, 
                  check_finite: bool = True,
                  bounds: Tuple[np.ndarray, np.ndarray] = (-np.inf, np.inf), 
                  method: Optional[str] = None, 
                  jac: Optional[Callable] = None,
                  data_mask: Optional[np.ndarray] = None, 
                  timeit: bool = False, 
                  **kwargs
                  ) -> Tuple[np.ndarray, np.ndarray]:
        """
    Use non-linear least squares to fit a function, f, to data.
    Assumes ``ydata = f(xdata, *params) + eps``.

    Parameters
    ----------
    f : callable
        The model function, f(x, ...). It must take the independent
        variable as the first argument and the parameters to fit as
        separate remaining arguments.
    xdata : array_like or object
        The independent variable where the data is measured.
        Should usually be an M-length sequence or an (k,M)-shaped array for
        functions with k predictors, but can actually be any object.
    ydata : array_like
        The dependent data, a length M array - nominally ``f(xdata, ...)``.
    p0 : array_like, optional
        Initial guess for the parameters (length N). If None, then the
        initial values will all be 1 (if the number of parameters for the
        function can be determined using introspection, otherwise a
        ValueError is raised).
    sigma : None or M-length sequence or MxM array, optional
        Determines the uncertainty in `ydata`. If we define residuals as
        ``r = ydata - f(xdata, *popt)``, then the interpretation of `sigma`
        depends on its number of dimensions:
            - A 1-D `sigma` should contain values of standard deviations of
              errors in `ydata`. In this case, the optimized function is
              ``chisq = sum((r / sigma) ** 2)``.
            - A 2-D `sigma` should contain the covariance matrix of
              errors in `ydata`. In this case, the optimized function is
              ``chisq = r.T @ inv(sigma) @ r``.
              .. versionadded:: 0.19
        None (default) is equivalent of 1-D `sigma` filled with ones.
    absolute_sigma : bool, optional
        If True, `sigma` is used in an absolute sense and the estimated parameter
        covariance `pcov` reflects these absolute values.
        If False (default), only the relative magnitudes of the `sigma` values matter.
        The returned parameter covariance matrix `pcov` is based on scaling
        `sigma` by a constant factor. This constant is set by demanding that the
        reduced `chisq` for the optimal parameters `popt` when using the
        *scaled* `sigma` equals unity. In other words, `sigma` is scaled to
        match the sample variance of the residuals after the fit. Default is False.
        Mathematically,
        ``pcov(absolute_sigma=False) = pcov(absolute_sigma=True) * chisq(popt)/(M-N)``
    check_finite : bool, optional
        If True, check that the input arrays do not contain nans of infs,
        and raise a ValueError if they do. Setting this parameter to
        False may silently produce nonsensical results if the input arrays
        do contain nans. Default is True.
    bounds : 2-tuple of array_like, optional
        Lower and upper bounds on parameters. Defaults to no bounds.
        Each element of the tuple must be either an array with the length equal
        to the number of parameters, or a scalar (in which case the bound is
        taken to be the same for all parameters). Use ``np.inf`` with an
        appropriate sign to disable bounds on all or some parameters.
        .. versionadded:: 0.17
    method : {'trf'}, optional
        Method to use for optimization. See `least_squares` for more details.
        Currently only 'trf' is implemented.
        .. versionadded:: 0.17
    jac : callable, string or None, optional
        Function with signature ``jac(x, ...)`` which computes the Jacobian
        matrix of the model function with respect to parameters as a dense
        array_like structure. It will be scaled according to provided `sigma`.
        If None (default), the Jacobian will be determined using JAX's automatic
        differentiation (AD) capabilities. We recommend not using an analytical
        Jacobian, as it is usually faster to use AD.
    kwargs
        Keyword arguments passed to `leastsq` for ``method='lm'`` or
        `least_squares` otherwise.
    Returns
    -------
    popt : array
        Optimal values for the parameters so that the sum of the squared
        residuals of ``f(xdata, *popt) - ydata`` is minimized.
    pcov : 2-D array
        The estimated covariance of popt. The diagonals provide the variance
        of the parameter estimate. To compute one standard deviation errors
        on the parameters use ``perr = np.sqrt(np.diag(pcov))``.
        How the `sigma` parameter affects the estimated covariance
        depends on `absolute_sigma` argument, as described above.
        If the Jacobian matrix at the solution doesn't have a full rank, then
        'lm' method returns a matrix filled with ``np.inf``, on the other hand
        'trf'  and 'dogbox' methods use Moore-Penrose pseudoinverse to compute
        the covariance matrix.
    Raises
    ------
    ValueError
        if either `ydata` or `xdata` contain NaNs, or if incompatible options
        are used.
    RuntimeError
        if the least-squares minimization fails.
    OptimizeWarning
        if covariance of the parameters can not be estimated.
    See Also
    --------
    least_squares : Minimize the sum of squares of nonlinear functions.
    
    Notes
    -----
    Refer to the docstring of `least_squares` for more information.
    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import jax.numpy as jnp
    >>> from jaxfit import CurveFit
    >>> def func(x, a, b, c):
    ...     return a * jnp.exp(-b * x) + c
    Define the data to be fit with some noise:
    >>> xdata = np.linspace(0, 4, 50)
    >>> y = func(xdata, 2.5, 1.3, 0.5)
    >>> rng = np.random.default_rng()
    >>> y_noise = 0.2 * rng.normal(size=xdata.size)
    >>> ydata = y + y_noise
    >>> plt.plot(xdata, ydata, 'b-', label='data')
    Fit for the parameters a, b, c of the function `func`:
    >>> cf = CurveFit()
    >>> popt, pcov = cf.curve_fit(func, xdata, ydata)
    >>> popt
    array([2.56274217, 1.37268521, 0.47427475])
    >>> plt.plot(xdata, func(xdata, *popt), 'r-',
    ...          label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    Constrain the optimization to the region of ``0 <= a <= 3``,
    ``0 <= b <= 1`` and ``0 <= c <= 0.5``:
    >>> cf = CurveFit()
    >>> popt, pcov = cf.curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5]))
    >>> popt
    array([2.43736712, 1.        , 0.34463856])
    >>> plt.plot(xdata, func(xdata, *popt), 'g--',
    ...          label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    >>> plt.xlabel('x')
    >>> plt.ylabel('y')
    >>> plt.legend()
    >>> plt.show()
    """
        
        if p0 is None:
            # determine number of parameters by inspecting the function
            sig = signature(f)
            args = sig.parameters
            if len(args) < 2:
                raise ValueError("Unable to determine number of fit parameters.")
            n = len(args) - 1
        else:
            p0 = np.atleast_1d(p0)
            n = p0.size
                    
        lb, ub = prepare_bounds(bounds, n)
        if p0 is None:
            p0 = _initialize_feasible(lb, ub)
            
        if method is None:
            method = 'trf'

        # NaNs cannot be handled
        if check_finite:
            ydata = np.asarray_chkfinite(ydata, float)
        else:
            ydata = np.asarray(ydata, float)
    
        if isinstance(xdata, (list, tuple, np.ndarray)):
            #should we be able to pass jax arrays
            # `xdata` is passed straight to the user-defined `f`, so allow
            # non-array_like `xdata`.
            if check_finite:
                xdata = np.asarray_chkfinite(xdata, float)
            else:
                xdata = np.asarray(xdata, float)
        else:
            raise ValueError('X needs arrays')
    
        if ydata.size == 0:
            raise ValueError("`ydata` must not be empty!")
            
        m = len(ydata)            
        xdims = xdata.ndim
        if xdims == 1:
            xlen = len(xdata)
        else:
            xlen = len(xdata[0])
        if xlen != m:
            print(xdata.shape, ydata.shape)
            raise ValueError('X and Y data lengths dont match')
            
            
        if self.flength is not None:
            len_diff = self.flength - m
            if data_mask is not None:
                if len(data_mask) != m:
                    raise ValueError('Data mask doesnt match data lengths.')
            else:
                data_mask = np.ones(m)
                if len_diff > 0:
                    data_mask = np.concatenate([data_mask, np.zeros(len_diff)])
        else:
            len_diff = 0
            data_mask = np.ones(m)
            
        
        if self.flength is not None:
            if len_diff >= 0:
                xdata, ydata = self.pad_fit_data(xdata, ydata, xdims, len_diff)
            else:
                print('Data length greater than fixed length. This means retracing will occur')
            
                   # Determine type of sigma
        if sigma is not None:   
            if not isinstance(sigma, np.ndarray):
                raise ValueError('Sigma must be numpy array.')
            # if 1-D, sigma are errors, define transform = 1/sigma
            ysize = ydata.size - len_diff
            if sigma.shape == (ysize, ):
                if len_diff > 0:
                    sigma = np.concatenate([sigma, np.ones([len_diff])])
                transform = self.sigma_transform1d(sigma, data_mask)
            # if 2-D, sigma is the covariance matrix,
            # define transform = L such that L L^T = C
            elif sigma.shape == (ysize, ysize):
                try:
                    if len_diff >= 0:
                        sigma_padded = np.identity(m + len_diff)
                        sigma_padded[:m,:m] = sigma
                        sigma = sigma_padded
                    # scipy.linalg.cholesky requires lower=True to return L L^T = A
                    transform = self.sigma_transform2d(sigma, data_mask)
                except:
                    raise ValueError("Probably:`sigma` must be positive definite.")
            else:
                print('sigma shape', sigma.shape)
                print('y shape', ydata.shape, ydata.size)
                print(len_diff)
                raise ValueError("`sigma` has incorrect shape.")
        else:
            transform = None
            

        if 'args' in kwargs:
            # The specification for the model function `f` does not support
            # additional arguments. Refer to the `curve_fit` docstring for
            # acceptable call signatures of `f`.
            raise ValueError("'args' is not a supported keyword argument.")
    
        if 'max_nfev' not in kwargs:
            kwargs['max_nfev'] = kwargs.pop('maxfev', None)
        
        st = time.time()
        if timeit:
            xdata = jnp.array(np.copy(xdata)).block_until_ready()
            ydata = jnp.array(np.copy(ydata)).block_until_ready()
        else:
            xdata = jnp.array(np.copy(xdata))
            ydata = jnp.array(np.copy(ydata))
        ctime = time.time() - st

        data_mask = jnp.array(data_mask, dtype=bool)
        res = self.ls.least_squares(f, p0, jac=jac, xdata=xdata, ydata=ydata, 
                                    data_mask=data_mask, transform=transform,
                                    bounds=bounds, method=method,
                                    timeit=timeit, **kwargs)

        if not res.success:
            raise RuntimeError("Optimal parameters not found: " + res.message)
        popt = res.x

        st = time.time()
        # ysize = len(res.fun)
        ysize = m
        cost = 2 * res.cost  # res.cost is half sum of squares!

        # Do Moore-Penrose inverse discarding zero singular values.
        # _, s, VT = svd(res.jac, full_matrices=False)
        outputs = self.covariance_svd(res.jac)
        s, VT = [np.array(output) for output in outputs]
        threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
        s = s[s > threshold]
        VT = VT[:s.size]
        pcov = np.dot(VT.T / s**2, VT)
        return_full = False
    
        warn_cov = False
        if pcov is None:
            # indeterminate covariance
            pcov = zeros((len(popt), len(popt)), dtype=float)
            pcov.fill(inf)
            warn_cov = True
        elif not absolute_sigma:
            if ysize > p0.size:
                s_sq = cost / (ysize - p0.size)
                pcov = pcov * s_sq
            else:
                pcov.fill(inf)
                warn_cov = True
    
        if warn_cov:
            warnings.warn('Covariance of the parameters could not be estimated',
                          category=OptimizeWarning)
        res.pop('jac')
        res.pop('fun')
        # self.res = res
        post_time = time.time() - st
        if return_full:
            raise RuntimeError("Return full only works for LM")
            # return popt, pcov, infodict, errmsg, ier
        elif timeit:
            return popt, pcov, res, post_time, ctime
        else:
            return popt, pcov

            

    