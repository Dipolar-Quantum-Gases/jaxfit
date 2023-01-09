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
        """Solve a nonlinear least-squares problem with bounds on the variables.
        Given the residuals f(x) (an m-D real function of n real
        variables) and the loss function rho(s) (a scalar function), `least_squares`
        finds a local minimum of the cost function F(x)::
            minimize F(x) = 0.5 * sum(rho(f_i(x)**2), i = 0, ..., m - 1)
            subject to lb <= x <= ub
        The purpose of the loss function rho(s) is to reduce the influence of
        outliers on the solution.

        Parameters
        ----------
        fun : callable
            Function which computes the vector of residuals, with the signature
            ``fun(x, *args, **kwargs)``, i.e., the minimization proceeds with
            respect to its first argument. The argument ``x`` passed to this
            function is an ndarray of shape (n,) (never a scalar, even for n=1).
            It must allocate and return a 1-D array_like of shape (m,) or a scalar.
            If the argument ``x`` is complex or the function ``fun`` returns
            complex residuals, it must be wrapped in a real function of real
            arguments, as shown at the end of the Examples section.
        x0 : array_like with shape (n,) or float
            Initial guess on independent variables. If float, it will be treated
            as a 1-D array with one element.
        jac : {None, callable}, optional
            Method of computing the Jacobian matrix (an m-by-n matrix, where
            element (i, j) is the partial derivative of f[i] with respect to
            x[j]). None uses JAX's jacfwd function and performs automatic 
            differentiation. The user can also pass in a callable function, it is 
            used as ``jac(x, *args, **kwargs)`` and should return a good approximation
            (or the exact value) for the Jacobian as an array_like (np.atleast_2d
            is applied).
        bounds : 2-tuple of array_like, optional
            Lower and upper bounds on independent variables. Defaults to no bounds.
            Each array must match the size of `x0` or be a scalar, in the latter
            case a bound will be the same for all variables. Use ``np.inf`` with
            an appropriate sign to disable bounds on all or some variables.
        method : {'trf'}, optional
            Algorithm to perform minimization, currently only 'trf' is supported.
                * 'trf' : Trust Region Reflective algorithm, particularly suitable
                for large sparse problems with bounds. Generally robust method.
            Default is 'trf'. See Notes for more information.
        ftol : float or None, optional
            Tolerance for termination by the change of the cost function. Default
            is 1e-8. The optimization process is stopped when ``dF < ftol * F``,
            and there was an adequate agreement between a local quadratic model and
            the true model in the last step.
            If None and 'method' is not 'lm', the termination by this condition is
            disabled. 
        xtol : float or None, optional
            Tolerance for termination by the change of the independent variables.
            Default is 1e-8. The exact condition depends on the `method` used:
                * For 'trf' and 'dogbox' : ``norm(dx) < xtol * (xtol + norm(x))``.
                * For 'lm' : ``Delta < xtol * norm(xs)``, where ``Delta`` is
                a trust-region radius and ``xs`` is the value of ``x``
                scaled according to `x_scale` parameter (see below).
            
            If None and 'method' is not 'lm', the termination by this condition is
            disabled. If 'method' is 'lm', this tolerance must be higher than
            machine epsilon.
        gtol : float or None, optional
            Tolerance for termination by the norm of the gradient. Default is 1e-8.
            The exact condition depends on a `method` used:
                * For 'trf' : ``norm(g_scaled, ord=np.inf) < gtol``, where
                ``g_scaled`` is the value of the gradient scaled to account for
                the presence of the bounds [STIR]_.
                * For 'dogbox' : ``norm(g_free, ord=np.inf) < gtol``, where
                ``g_free`` is the gradient with respect to the variables which
                are not in the optimal state on the boundary.
                * For 'lm' : the maximum absolute value of the cosine of angles
                between columns of the Jacobian and the residual vector is less
                than `gtol`, or the residual vector is zero.
            
            If None and 'method' is not 'lm', the termination by this condition is
            disabled. If 'method' is 'lm', this tolerance must be higher than
            machine epsilon.
        x_scale : array_like or 'jac', optional
            Characteristic scale of each variable. Setting `x_scale` is equivalent
            to reformulating the problem in scaled variables ``xs = x / x_scale``.
            An alternative view is that the size of a trust region along jth
            dimension is proportional to ``x_scale[j]``. Improved convergence may
            be achieved by setting `x_scale` such that a step of a given size
            along any of the scaled variables has a similar effect on the cost
            function. If set to 'jac', the scale is iteratively updated using the
            inverse norms of the columns of the Jacobian matrix (as described in
            [JJMore]_).
        loss : str or callable, optional
            Determines the loss function. The following keyword values are allowed:
                * 'linear' (default) : ``rho(z) = z``. Gives a standard
                least-squares problem.
                * 'soft_l1' : ``rho(z) = 2 * ((1 + z)**0.5 - 1)``. The smooth
                approximation of l1 (absolute value) loss. Usually a good
                choice for robust least squares.
                * 'huber' : ``rho(z) = z if z <= 1 else 2*z**0.5 - 1``. Works
                similarly to 'soft_l1'.
                * 'cauchy' : ``rho(z) = ln(1 + z)``. Severely weakens outliers
                influence, but may cause difficulties in optimization process.
                * 'arctan' : ``rho(z) = arctan(z)``. Limits a maximum loss on
                a single residual, has properties similar to 'cauchy'.

            If callable, it must take a 1-D ndarray ``z=f**2`` and return an
            array_like with shape (3, m) where row 0 contains function values,
            row 1 contains first derivatives and row 2 contains second
            derivatives. Method 'lm' supports only 'linear' loss.
        f_scale : float, optional
            Value of soft margin between inlier and outlier residuals, default
            is 1.0. The loss function is evaluated as follows
            ``rho_(f**2) = C**2 * rho(f**2 / C**2)``, where ``C`` is `f_scale`,
            and ``rho`` is determined by `loss` parameter. This parameter has
            no effect with ``loss='linear'``, but for other `loss` values it is
            of crucial importance.
        max_nfev : None or int, optional
            Maximum number of function evaluations before the termination.
            If None (default), the value is chosen automatically:
                * For 'trf' and 'dogbox' : 100 * n.
                * For 'lm' :  100 * n if `jac` is callable and 100 * n * (n + 1)
                otherwise (because 'lm' counts function calls in Jacobian
                estimation).

        diff_step : None or array_like, optional
            Determines the relative step size for the finite difference
            approximation of the Jacobian. The actual step is computed as
            ``x * diff_step``. If None (default), then `diff_step` is taken to be
            a conventional "optimal" power of machine epsilon for the finite
            difference scheme used [NR]_.
        tr_solver : {None, 'exact', 'lsmr'}, optional
            Method for solving trust-region subproblems, relevant only for 'trf'
            and 'dogbox' methods.
                * 'exact' is suitable for not very large problems with dense
                Jacobian matrices. The computational complexity per iteration is
                comparable to a singular value decomposition of the Jacobian
                matrix.
                * 'lsmr' is suitable for problems with sparse and large Jacobian
                matrices. It uses the iterative procedure
                `scipy.sparse.linalg.lsmr` for finding a solution of a linear
                least-squares problem and only requires matrix-vector product
                evaluations.

            If None (default), the solver is chosen based on the type of Jacobian
            returned on the first iteration.
        tr_options : dict, optional
            Keyword options passed to trust-region solver.
                * ``tr_solver='exact'``: `tr_options` are ignored.
                * ``tr_solver='lsmr'``: options for `scipy.sparse.linalg.lsmr`.
                Additionally,  ``method='trf'`` supports  'regularize' option
                (bool, default is True), which adds a regularization term to the
                normal equation, which improves convergence if the Jacobian is
                rank-deficient [Byrd]_ (eq. 3.4).

        jac_sparsity : {None, array_like, sparse matrix}, optional
            Defines the sparsity structure of the Jacobian matrix for finite
            difference estimation, its shape must be (m, n). If the Jacobian has
            only few non-zero elements in *each* row, providing the sparsity
            structure will greatly speed up the computations [Curtis]_. A zero
            entry means that a corresponding element in the Jacobian is identically
            zero. If provided, forces the use of 'lsmr' trust-region solver.
            If None (default), then dense differencing will be used. Has no effect
            for 'lm' method.
        verbose : {0, 1, 2}, optional
            Level of algorithm's verbosity:
                * 0 (default) : work silently.
                * 1 : display a termination report.
                * 2 : display progress during iterations (not supported by 'lm'
                method).

        args, kwargs : tuple and dict, optional
            Additional arguments passed to `fun` and `jac`. Both empty by default.
            The calling signature is ``fun(x, *args, **kwargs)`` and the same for
            `jac`.
        Returns
        -------
        result : OptimizeResult
            `OptimizeResult` with the following fields defined:
                x : ndarray, shape (n,)
                    Solution found.
                cost : float
                    Value of the cost function at the solution.
                fun : ndarray, shape (m,)
                    Vector of residuals at the solution.
                jac : ndarray, sparse matrix or LinearOperator, shape (m, n)
                    Modified Jacobian matrix at the solution, in the sense that J^T J
                    is a Gauss-Newton approximation of the Hessian of the cost function.
                    The type is the same as the one used by the algorithm.
                grad : ndarray, shape (m,)
                    Gradient of the cost function at the solution.
                optimality : float
                    First-order optimality measure. In unconstrained problems, it is
                    always the uniform norm of the gradient. In constrained problems,
                    it is the quantity which was compared with `gtol` during iterations.
                active_mask : ndarray of int, shape (n,)
                    Each component shows whether a corresponding constraint is active
                    (that is, whether a variable is at the bound):
                        *  0 : a constraint is not active.
                        * -1 : a lower bound is active.
                        *  1 : an upper bound is active.

                    Might be somewhat arbitrary for 'trf' method as it generates a
                    sequence of strictly feasible iterates and `active_mask` is
                    determined within a tolerance threshold.
                nfev : int
                    Number of function evaluations done. Methods 'trf' and 'dogbox' do
                    not count function calls for numerical Jacobian approximation, as
                    opposed to 'lm' method.
                njev : int or None
                    Number of Jacobian evaluations done. If numerical Jacobian
                    approximation is used in 'lm' method, it is set to None.
                status : int
                    The reason for algorithm termination:
                        * -1 : improper input parameters status returned from MINPACK.
                        *  0 : the maximum number of function evaluations is exceeded.
                        *  1 : `gtol` termination condition is satisfied.
                        *  2 : `ftol` termination condition is satisfied.
                        *  3 : `xtol` termination condition is satisfied.
                        *  4 : Both `ftol` and `xtol` termination conditions are satisfied.

                message : str
                    Verbal description of the termination reason.
                success : bool
                    True if one of the convergence criteria is satisfied (`status` > 0).
        See Also
        --------
        leastsq : A legacy wrapper for the MINPACK implementation of the
                Levenberg-Marquadt algorithm.
        curve_fit : Least-squares minimization applied to a curve-fitting problem.
        Notes
        -----
        Method 'lm' (Levenberg-Marquardt) calls a wrapper over least-squares
        algorithms implemented in MINPACK (lmder, lmdif). It runs the
        Levenberg-Marquardt algorithm formulated as a trust-region type algorithm.
        The implementation is based on paper [JJMore]_, it is very robust and
        efficient with a lot of smart tricks. It should be your first choice
        for unconstrained problems. Note that it doesn't support bounds. Also,
        it doesn't work when m < n.
        Method 'trf' (Trust Region Reflective) is motivated by the process of
        solving a system of equations, which constitute the first-order optimality
        condition for a bound-constrained minimization problem as formulated in
        [STIR]_. The algorithm iteratively solves trust-region subproblems
        augmented by a special diagonal quadratic term and with trust-region shape
        determined by the distance from the bounds and the direction of the
        gradient. This enhancements help to avoid making steps directly into bounds
        and efficiently explore the whole space of variables. To further improve
        convergence, the algorithm considers search directions reflected from the
        bounds. To obey theoretical requirements, the algorithm keeps iterates
        strictly feasible. With dense Jacobians trust-region subproblems are
        solved by an exact method very similar to the one described in [JJMore]_
        (and implemented in MINPACK). The difference from the MINPACK
        implementation is that a singular value decomposition of a Jacobian
        matrix is done once per iteration, instead of a QR decomposition and series
        of Givens rotation eliminations. For large sparse Jacobians a 2-D subspace
        approach of solving trust-region subproblems is used [STIR]_, [Byrd]_.
        The subspace is spanned by a scaled gradient and an approximate
        Gauss-Newton solution delivered by `scipy.sparse.linalg.lsmr`. When no
        constraints are imposed the algorithm is very similar to MINPACK and has
        generally comparable performance. The algorithm works quite robust in
        unbounded and bounded problems, thus it is chosen as a default algorithm.
        Method 'dogbox' operates in a trust-region framework, but considers
        rectangular trust regions as opposed to conventional ellipsoids [Voglis]_.
        The intersection of a current trust region and initial bounds is again
        rectangular, so on each iteration a quadratic minimization problem subject
        to bound constraints is solved approximately by Powell's dogleg method
        [NumOpt]_. The required Gauss-Newton step can be computed exactly for
        dense Jacobians or approximately by `scipy.sparse.linalg.lsmr` for large
        sparse Jacobians. The algorithm is likely to exhibit slow convergence when
        the rank of Jacobian is less than the number of variables. The algorithm
        often outperforms 'trf' in bounded problems with a small number of
        variables.
        Robust loss functions are implemented as described in [BA]_. The idea
        is to modify a residual vector and a Jacobian matrix on each iteration
        such that computed gradient and Gauss-Newton Hessian approximation match
        the true gradient and Hessian approximation of the cost function. Then
        the algorithm proceeds in a normal way, i.e., robust loss functions are
        implemented as a simple wrapper over standard least-squares algorithms.
        .. versionadded:: 0.17.0
        References
        ----------
        .. [STIR] M. A. Branch, T. F. Coleman, and Y. Li, "A Subspace, Interior,
                and Conjugate Gradient Method for Large-Scale Bound-Constrained
                Minimization Problems," SIAM Journal on Scientific Computing,
                Vol. 21, Number 1, pp 1-23, 1999.
        .. [NR] William H. Press et. al., "Numerical Recipes. The Art of Scientific
                Computing. 3rd edition", Sec. 5.7.
        .. [Byrd] R. H. Byrd, R. B. Schnabel and G. A. Shultz, "Approximate
                solution of the trust region problem by minimization over
                two-dimensional subspaces", Math. Programming, 40, pp. 247-263,
                1988.
        .. [Curtis] A. Curtis, M. J. D. Powell, and J. Reid, "On the estimation of
                    sparse Jacobian matrices", Journal of the Institute of
                    Mathematics and its Applications, 13, pp. 117-120, 1974.
        .. [JJMore] J. J. More, "The Levenberg-Marquardt Algorithm: Implementation
                    and Theory," Numerical Analysis, ed. G. A. Watson, Lecture
                    Notes in Mathematics 630, Springer Verlag, pp. 105-116, 1977.
        .. [Voglis] C. Voglis and I. E. Lagaris, "A Rectangular Trust Region
                    Dogleg Approach for Unconstrained and Bound Constrained
                    Nonlinear Optimization", WSEAS International Conference on
                    Applied Mathematics, Corfu, Greece, 2004.
        .. [NumOpt] J. Nocedal and S. J. Wright, "Numerical optimization,
                    2nd edition", Chapter 4.
        .. [BA] B. Triggs et. al., "Bundle Adjustment - A Modern Synthesis",
                Proceedings of the International Workshop on Vision Algorithms:
                Theory and Practice, pp. 298-372, 1999.
                
        Examples
        --------
        In this example we find a minimum of the Rosenbrock function without bounds
        on independent variables.
        >>> def fun_rosenbrock(x):
        ...     return np.array([10 * (x[1] - x[0]**2), (1 - x[0])])
        Notice that we only provide the vector of the residuals. The algorithm
        constructs the cost function as a sum of squares of the residuals, which
        gives the Rosenbrock function. The exact minimum is at ``x = [1.0, 1.0]``.
        >>> from scipy.optimize import least_squares
        >>> x0_rosenbrock = np.array([2, 2])
        >>> res_1 = least_squares(fun_rosenbrock, x0_rosenbrock)
        >>> res_1.x
        array([ 1.,  1.])
        >>> res_1.cost
        9.8669242910846867e-30
        >>> res_1.optimality
        8.8928864934219529e-14
        We now constrain the variables, in such a way that the previous solution
        becomes infeasible. Specifically, we require that ``x[1] >= 1.5``, and
        ``x[0]`` left unconstrained. To this end, we specify the `bounds` parameter
        to `least_squares` in the form ``bounds=([-np.inf, 1.5], np.inf)``.
        We also provide the analytic Jacobian:
        >>> def jac_rosenbrock(x):
        ...     return np.array([
        ...         [-20 * x[0], 10],
        ...         [-1, 0]])
        Putting this all together, we see that the new solution lies on the bound:
        >>> res_2 = least_squares(fun_rosenbrock, x0_rosenbrock, jac_rosenbrock,
        ...                       bounds=([-np.inf, 1.5], np.inf))
        >>> res_2.x
        array([ 1.22437075,  1.5       ])
        >>> res_2.cost
        0.025213093946805685
        >>> res_2.optimality
        1.5885401433157753e-07
        Now we solve a system of equations (i.e., the cost function should be zero
        at a minimum) for a Broyden tridiagonal vector-valued function of 100000
        variables:
        >>> def fun_broyden(x):
        ...     f = (3 - x) * x + 1
        ...     f[1:] -= x[:-1]
        ...     f[:-1] -= 2 * x[1:]
        ...     return f
        The corresponding Jacobian matrix is sparse. We tell the algorithm to
        estimate it by finite differences and provide the sparsity structure of
        Jacobian to significantly speed up this process.
        >>> from scipy.sparse import lil_matrix
        >>> def sparsity_broyden(n):
        ...     sparsity = lil_matrix((n, n), dtype=int)
        ...     i = np.arange(n)
        ...     sparsity[i, i] = 1
        ...     i = np.arange(1, n)
        ...     sparsity[i, i - 1] = 1
        ...     i = np.arange(n - 1)
        ...     sparsity[i, i + 1] = 1
        ...     return sparsity
        ...
        >>> n = 100000
        >>> x0_broyden = -np.ones(n)
        ...
        >>> res_3 = least_squares(fun_broyden, x0_broyden,
        ...                       jac_sparsity=sparsity_broyden(n))
        >>> res_3.cost
        4.5687069299604613e-23
        >>> res_3.optimality
        1.1650454296851518e-11
        Let's also solve a curve fitting problem using robust loss function to
        take care of outliers in the data. Define the model function as
        ``y = a + b * exp(c * t)``, where t is a predictor variable, y is an
        observation and a, b, c are parameters to estimate.
        First, define the function which generates the data with noise and
        outliers, define the model parameters, and generate data:
        >>> from numpy.random import default_rng
        >>> rng = default_rng()
        >>> def gen_data(t, a, b, c, noise=0., n_outliers=0, seed=None):
        ...     rng = default_rng(seed)
        ...
        ...     y = a + b * np.exp(t * c)
        ...
        ...     error = noise * rng.standard_normal(t.size)
        ...     outliers = rng.integers(0, t.size, n_outliers)
        ...     error[outliers] *= 10
        ...
        ...     return y + error
        ...
        >>> a = 0.5
        >>> b = 2.0
        >>> c = -1
        >>> t_min = 0
        >>> t_max = 10
        >>> n_points = 15
        ...
        >>> t_train = np.linspace(t_min, t_max, n_points)
        >>> y_train = gen_data(t_train, a, b, c, noise=0.1, n_outliers=3)
        Define function for computing residuals and initial estimate of
        parameters.
        >>> def fun(x, t, y):
        ...     return x[0] + x[1] * np.exp(x[2] * t) - y
        ...
        >>> x0 = np.array([1.0, 1.0, 0.0])
        Compute a standard least-squares solution:
        >>> res_lsq = least_squares(fun, x0, args=(t_train, y_train))
        Now compute two solutions with two different robust loss functions. The
        parameter `f_scale` is set to 0.1, meaning that inlier residuals should
        not significantly exceed 0.1 (the noise level used).
        >>> res_soft_l1 = least_squares(fun, x0, loss='soft_l1', f_scale=0.1,
        ...                             args=(t_train, y_train))
        >>> res_log = least_squares(fun, x0, loss='cauchy', f_scale=0.1,
        ...                         args=(t_train, y_train))
        And, finally, plot all the curves. We see that by selecting an appropriate
        `loss`  we can get estimates close to optimal even in the presence of
        strong outliers. But keep in mind that generally it is recommended to try
        'soft_l1' or 'huber' losses first (if at all necessary) as the other two
        options may cause difficulties in optimization process.
        >>> t_test = np.linspace(t_min, t_max, n_points * 10)
        >>> y_true = gen_data(t_test, a, b, c)
        >>> y_lsq = gen_data(t_test, *res_lsq.x)
        >>> y_soft_l1 = gen_data(t_test, *res_soft_l1.x)
        >>> y_log = gen_data(t_test, *res_log.x)
        ...
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(t_train, y_train, 'o')
        >>> plt.plot(t_test, y_true, 'k', linewidth=2, label='true')
        >>> plt.plot(t_test, y_lsq, label='linear loss')
        >>> plt.plot(t_test, y_soft_l1, label='soft_l1 loss')
        >>> plt.plot(t_test, y_log, label='cauchy loss')
        >>> plt.xlabel("t")
        >>> plt.ylabel("y")
        >>> plt.legend()
        >>> plt.show()
        In the next example, we show how complex-valued residual functions of
        complex variables can be optimized with ``least_squares()``. Consider the
        following function:
        >>> def f(z):
        ...     return z - (0.5 + 0.5j)
        We wrap it into a function of real variables that returns real residuals
        by simply handling the real and imaginary parts as independent variables:
        >>> def f_wrap(x):
        ...     fx = f(x[0] + 1j*x[1])
        ...     return np.array([fx.real, fx.imag])
        Thus, instead of the original m-D complex function of n complex
        variables we optimize a 2m-D real function of 2n real variables:
        >>> from scipy.optimize import least_squares
        >>> res_wrapped = least_squares(f_wrap, (0.1, 0.1), bounds=([0, 0], [1, 1]))
        >>> z = res_wrapped.x[0] + res_wrapped.x[1]*1j
        >>> z
        (0.49999999999925893+0.49999999999925893j)
        """
        
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