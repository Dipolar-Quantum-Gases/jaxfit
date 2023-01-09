"""Trust Region Reflective algorithm for least-squares optimization.
The algorithm is based on ideas from paper [STIR]_. The main idea is to
account for the presence of the bounds by appropriate scaling of the variables (or,
equivalently, changing a trust-region shape). Let's introduce a vector v:
           | ub[i] - x[i], if g[i] < 0 and ub[i] < np.inf
    v[i] = | x[i] - lb[i], if g[i] > 0 and lb[i] > -np.inf
           | 1,           otherwise
where g is the gradient of a cost function and lb, ub are the bounds. Its
components are distances to the bounds at which the anti-gradient points (if
this distance is finite). Define a scaling matrix D = diag(v**0.5).
First-order optimality conditions can be stated as
    D^2 g(x) = 0.
Meaning that components of the gradient should be zero for strictly interior
variables, and components must point inside the feasible region for variables
on the bound.
Now consider this system of equations as a new optimization problem. If the
point x is strictly interior (not on the bound), then the left-hand side is
differentiable and the Newton step for it satisfies
    (D^2 H + diag(g) Jv) p = -D^2 g
where H is the Hessian matrix (or its J^T J approximation in least squares),
Jv is the Jacobian matrix of v with components -1, 1 or 0, such that all
elements of matrix C = diag(g) Jv are non-negative. Introduce the change
of the variables x = D x_h (_h would be "hat" in LaTeX). In the new variables,
we have a Newton step satisfying
    B_h p_h = -g_h,
where B_h = D H D + C, g_h = D g. In least squares B_h = J_h^T J_h, where
J_h = J D. Note that J_h and g_h are proper Jacobian and gradient with respect
to "hat" variables. To guarantee global convergence we formulate a
trust-region problem based on the Newton step in the new variables:
    0.5 * p_h^T B_h p + g_h^T p_h -> min, ||p_h|| <= Delta
In the original space B = H + D^{-1} C D^{-1}, and the equivalent trust-region
problem is
    0.5 * p^T B p + g^T p -> min, ||D^{-1} p|| <= Delta
Here, the meaning of the matrix D becomes more clear: it alters the shape
of a trust-region, such that large steps towards the bounds are not allowed.
In the implementation, the trust-region problem is solved in "hat" space,
but handling of the bounds is done in the original space (see below and read
the code).
The introduction of the matrix D doesn't allow to ignore bounds, the algorithm
must keep iterates strictly feasible (to satisfy aforementioned
differentiability), the parameter theta controls step back from the boundary
(see the code for details).
The algorithm does another important trick. If the trust-region solution
doesn't fit into the bounds, then a reflected (from a firstly encountered
bound) search direction is considered. For motivation and analysis refer to
[STIR]_ paper (and other papers of the authors). In practice, it doesn't need
a lot of justifications, the algorithm simply chooses the best step among
three: a constrained trust-region step, a reflected step and a constrained
Cauchy step (a minimizer along -g_h in "hat" space, or -D^2 g in the original
space).
Another feature is that a trust-region radius control strategy is modified to
account for appearance of the diagonal C matrix (called diag_h in the code).
Note that all described peculiarities are completely gone as we consider
problems without bounds (the algorithm becomes a standard trust-region type
algorithm very similar to ones implemented in MINPACK).
The implementation supports two methods of solving the trust-region problem.
The first, called 'exact', applies SVD on Jacobian and then solves the problem
very accurately using the algorithm described in [JJMore]_. It is not
applicable to large problem. The second, called 'lsmr', uses the 2-D subspace
approach (sometimes called "indefinite dogleg"), where the problem is solved
in a subspace spanned by the gradient and the approximate Gauss-Newton step
found by ``scipy.sparse.linalg.lsmr``. A 2-D trust-region problem is
reformulated as a 4th order algebraic equation and solved very accurately by
``numpy.roots``. The subspace approach allows to solve very large problems
(up to couple of millions of residuals on a regular PC), provided the Jacobian
matrix is sufficiently sparse.
References
----------
.. [STIR] Branch, M.A., T.F. Coleman, and Y. Li, "A Subspace, Interior,
      and Conjugate Gradient Method for Large-Scale Bound-Constrained
      Minimization Problems," SIAM Journal on Scientific Computing,
      Vol. 21, Number 1, pp 1-23, 1999.
.. [JJMore] More, J. J., "The Levenberg-Marquardt Algorithm: Implementation
    and Theory," Numerical Analysis, ed. G. A. Watson, Lecture
"""
import numpy as np
from numpy.linalg import norm
import time
from typing import Callable, Optional, Tuple, Union, List, Dict, Any, Sequence

from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.linalg import svd as jax_svd
from jax import jit
from jax.tree_util import tree_flatten

from jaxfit.common_scipy import (update_tr_radius, solve_lsq_trust_region, 
                    check_termination, CL_scaling_vector,
                    make_strictly_feasible, find_active_constraints, in_bounds, 
                    step_size_to_bound, intersect_trust_region, 
                    minimize_quadratic_1d, print_header_nonlinear,
                    print_iteration_nonlinear)

from jaxfit.common_jax import CommonJIT
from jaxfit._optimize import OptimizeResult


class TrustRegionJITFunctions():
    """JIT functions for trust region algorithm."""
    
    def __init__(self):
        """Call all of the create functions which create the JAX/JIT functions
        that are members of the class."""
        self.create_grad_func()
        self.create_grad_hat()
        self.create_svd_funcs()
        self.create_default_loss_func()
        self.create_calculate_cost()
        self.create_check_isfinite()


    def create_default_loss_func(self):
        """Create the default loss function which is simply the sum of the
        squares of the residuals."""
        @jit
        def loss_function(f: jnp.ndarray) -> jnp.ndarray:
            """The default loss function is the sum of the squares of the
            residuals divided by two.

            Parameters
            ----------
            f : jnp.ndarray
                The residuals.

            Returns
            -------
            jnp.ndarray
                The loss function value.
            """
            return 0.5 * jnp.dot(f, f)
            # return 0.5 * jnp.sum(f**2)
            
        self.default_loss_func = loss_function

    
    def create_grad_func(self):
        """Create the function to compute the gradient of the loss function
        which is simply the function evaluation dotted with the Jacobian."""
        @jit
        def compute_grad(J: jnp.ndarray, f: jnp.ndarray) -> jnp.ndarray:
            """Compute the gradient of the loss function.

            Parameters
            ----------
            J : jnp.ndarray
                The Jacobian matrix.
            f : jnp.ndarray
                The residuals.

            Returns
            -------
            jnp.ndarray
                The gradient of the loss function.
            """
            return f.dot(J)
        self.compute_grad = compute_grad


    def create_grad_hat(self):
        """Calculate the gradient in the "hat" space, which is just multiplying
        the gradient by the diagonal matrix D. This is used in the trust region
        algorithm. Here we only use the diagonals of D, since D is diagonal.
        """
        @jit
        def compute_grad_hat(g: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
            """Compute the gradient in the "hat" space.
            
            Parameters
            ----------
            g : jnp.ndarray
            The gradient of the loss function.
            d : jnp.ndarray
            The diagonal of the diagonal matrix D.
            Returns
            -------
            jnp.ndarray
            The gradient in the "hat" space.
            """
            return d * g
        self.compute_grad_hat = compute_grad_hat

    
    def create_svd_funcs(self):
        """Create the functions to compute the SVD of the Jacobian matrix.
        There are two versions, one for problems with bounds and one for
        problems without bounds. The version for problems with bounds is
        slightly more complicated."""

        @jit
        def svd_no_bounds(J: jnp.ndarray, 
                          d: jnp.ndarray, 
                          f: jnp.ndarray
                          ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, 
                                     jnp.ndarray, jnp.ndarray]:
            """Compute the SVD of the Jacobian matrix, J, in the "hat" space.
            This is the version for problems without bounds.

            Parameters
            ----------
            J : jnp.ndarray
                The Jacobian matrix.
            d : jnp.ndarray
                The diagonal of the diagonal matrix D.
            f : jnp.ndarray
                The residuals.

            Returns
            -------
            J_h : jnp.ndarray
                  the Jacobian matrix in the "hat" space.
            U : jnp.ndarray
                the left singular vectors of the SVD of J_h.
            s : jnp.ndarray
                the singular values of the SVD of J_h.
            V : jnp.ndarray
                the right singular vectors of the SVD of J_h.
            uf : jnp.ndarray
                 the dot product of U.T and f.
            """
            J_h = J * d
            U, s, V = jax_svd(J_h, full_matrices=False)
            V = V.T
            uf = U.T.dot(f)
            return J_h, U, s, V, uf
    
        @jit
        def svd_bounds(f: jnp.ndarray,
                       J: jnp.ndarray,
                       d: jnp.ndarray,
                       J_diag: jnp.ndarray,
                       f_zeros: jnp.ndarray
                       ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, 
                                  jnp.ndarray, jnp.ndarray]:
            """Compute the SVD of the Jacobian matrix, J, in the "hat" space.
            This is the version for problems with bounds.
            Parameters
            ----------
            J : jnp.ndarray
                The Jacobian matrix.
            d : jnp.ndarray
                The diagonal of the diagonal matrix D.
            f : jnp.ndarray
                The residuals.
            J_diag : jnp.ndarray
                    Added term to Jacobian matrix.
            f_zeros : jnp.ndarray
                    Empty residuals for the added term.


            Returns
            -------
            J_h : jnp.ndarray
                  the Jacobian matrix in the "hat" space.
            U : jnp.ndarray
                the left singular vectors of the SVD of J_h.
            s : jnp.ndarray
                the singular values of the SVD of J_h.
            V : jnp.ndarray
                the right singular vectors of the SVD of J_h.
            uf : jnp.ndarray
                 the dot product of U.T and f.
            """
            J_h = J * d
            J_augmented = jnp.concatenate([J_h, J_diag])
            f_augmented = jnp.concatenate([f, f_zeros])
            U, s, V = jax_svd(J_augmented, full_matrices=False)
            V = V.T
            uf = U.T.dot(f_augmented)
            return J_h, U, s, V, uf
        
        self.svd_no_bounds = svd_no_bounds
        self.svd_bounds = svd_bounds
                
            
    def create_calculate_cost(self):
        """Create the function to calculate the cost function."""
        @jit
        def calculate_cost(rho, data_mask):
            """Calculate the cost function.
            Parameters
            ----------
            rho : jnp.ndarray
                The per element cost times two.
            data_mask : jnp.ndarray
                The mask for the data.
            Returns
            -------
            jnp.ndarray
                The cost function.
            """
            cost_array = jnp.where(data_mask, rho[0], 0)
            return 0.5 * jnp.sum(cost_array)
        self.calculate_cost = calculate_cost
    
    
    def create_check_isfinite(self):
        """Create the function to check if the evaluated residuals are finite."""
        @jit
        def isfinite(f_new: jnp.ndarray) -> bool:
            """Check if the evaluated residuals are finite.
            Parameters
            ----------
            f_new : jnp.ndarray
                The evaluated residuals.
            Returns
            -------
            bool
                True if all residuals are finite, False otherwise.
            """
            return jnp.all(jnp.isfinite(f_new))
        self.check_isfinite = isfinite


class TrustRegionReflective(TrustRegionJITFunctions):
    
    def __init__(self):
        """Initialize the TrustRegionReflective class."""
        super().__init__()
        self.cJIT = CommonJIT()


    def trf(self, 
            fun: Callable, 
            xdata: Union[jnp.ndarray, Tuple[jnp.ndarray]], 
            ydata: jnp.ndarray, 
            jac: Callable,
            data_mask: jnp.ndarray, 
            transform: jnp.ndarray, 
            x0: np.ndarray, 
            f0: jnp.ndarray, 
            J0: jnp.ndarray,
            lb: np.ndarray, 
            ub: np.ndarray, 
            ftol: float, 
            xtol: float, 
            gtol: float, 
            max_nfev: int, 
            f_scale: float, 
            x_scale: np.ndarray, 
            loss_function: Union[None, Callable],
            tr_options: Dict, 
            verbose: int, 
            timeit: bool=False
            ) -> Dict:
        """Minimize a scalar function of one or more variables using the
        trust-region reflective algorithm. Although I think this is not good
        coding style, I maintained the original code format from SciPy such
        that the code is easier to compare with the original. See the note
        from the algorithms original author below.
        
        
        For efficiency, it makes sense to run 
        the simplified version of the algorithm when no bounds are imposed. 
        We decided to write the two separate functions. It violates the DRY 
        principle, but the individual functions are kept the most readable.
        
        Parameters
        ----------
        fun : callable
            The residual function
        xdata : array_like or tuple of array_like
            The independent variable where the data is measured. If `xdata` is a
            tuple, then the input arguments to `fun` are assumed to be
            ``(xdata[0], xdata[1], ...)``.
        ydata : jnp.ndarray
            The dependent data
        jac : callable
            The Jacobian of `fun`.
        data_mask : jnp.ndarray
            The mask for the data.
        transform : jnp.ndarray
            The uncertainty transform for the data.
        x0 : jnp.ndarray
            Initial guess. Array of real elements of size (n,), where 'n' is the
            number of independent variables.
        f0 : jnp.ndarray
            Initial residuals. Array of real elements of size (m,), where 'm' is
            the number of data points.
        J0 : jnp.ndarray
            Initial Jacobian. Array of real elements of size (m, n), where 'm' is
            the number of data points and 'n' is the number of independent
            variables.
        lb : jnp.ndarray
            Lower bounds on independent variables. Array of real elements of size
            (n,), where 'n' is the number of independent variables.
        ub : jnp.ndarray
            Upper bounds on independent variables. Array of real elements of size
            (n,), where 'n' is the number of independent variables.
        ftol : float
            Tolerance for termination by the change of the cost function.
        xtol : float
            Tolerance for termination by the change of the independent variables.
        gtol : float
            Tolerance for termination by the norm of the gradient.
        max_nfev : int
            Maximum number of function evaluations.
        f_scale : float
            Cost function scalar
        x_scale : jnp.ndarray
            Scaling factors for independent variables.
        loss_function : callable, optional
            Loss function. If None, the standard least-squares problem is
            solved.
        tr_options : dict
            Options for the trust-region algorithm.
        verbose : int
            Level of algorithm's verbosity:
                * 0 (default) : work silently.
                * 1 : display a termination report.
        timeit : bool, optional
            If True, the time for each step is measured if the unbounded
            version is being ran. Default is False.
        """
        # bounded or unbounded version
        if np.all(lb == -np.inf) and np.all(ub == np.inf):
            # unbounded version as timed and untimed version
            if not timeit:
                return self.trf_no_bounds(fun, xdata, ydata, jac, data_mask, 
                                          transform, x0, f0, J0,
                lb, ub, ftol, xtol, gtol, max_nfev, f_scale, x_scale, 
                loss_function, tr_options, verbose)
            else:
                return self.trf_no_bounds_timed(fun, xdata, ydata, jac, data_mask, transform, x0, 
                f0, J0, lb, ub, ftol, xtol, gtol, max_nfev, f_scale, x_scale, 
                loss_function, tr_options, verbose)
        else:
            return self.trf_bounds(fun, xdata, ydata, jac, data_mask, transform, x0, 
                    f0, J0, lb, ub, ftol, xtol, gtol, max_nfev, f_scale, x_scale, 
                    loss_function, tr_options, verbose)
        

    def trf_no_bounds(self, 
                      fun: Callable, 
                      xdata: Union[jnp.ndarray, Tuple[jnp.ndarray]], 
                      ydata: jnp.ndarray, 
                      jac: Callable, 
                      data_mask: jnp.ndarray, 
                      transform: jnp.ndarray, 
                      x0: np.ndarray, 
                      f: jnp.ndarray, 
                      J: jnp.ndarray, 
                      lb: np.ndarray, 
                      ub: np.ndarray,    
                      ftol: float, 
                      xtol: float, 
                      gtol: float, 
                      max_nfev: int,
                      f_scale: float, 
                      x_scale: np.ndarray, 
                      loss_function: Union[None, Callable], 
                      tr_options: Dict, 
                      verbose: int
                      ) -> Dict:
        """Unbounded version of the trust-region reflective algorithm.

        Parameters
        ----------
        fun : callable
            The residual function
        xdata : array_like or tuple of array_like
            The independent variable where the data is measured. If `xdata` is a
            tuple, then the input arguments to `fun` are assumed to be
            ``(xdata[0], xdata[1], ...)``.
        ydata : jnp.ndarray
            The dependent data
        jac : callable
            The Jacobian of `fun`.
        data_mask : jnp.ndarray
            The mask for the data.
        transform : jnp.ndarray
            The uncertainty transform for the data.
        x0 : jnp.ndarray
            Initial guess. Array of real elements of size (n,), where 'n' is the
            number of independent variables.
        f0 : jnp.ndarray
            Initial residuals. Array of real elements of size (m,), where 'm' is
            the number of data points.
        J0 : jnp.ndarray
            Initial Jacobian. Array of real elements of size (m, n), where 'm' is
            the number of data points and 'n' is the number of independent
            variables.
        lb : jnp.ndarray
            Lower bounds on independent variables. Array of real elements of size
            (n,), where 'n' is the number of independent variables. 
        ub : jnp.ndarray
            Upper bounds on independent variables. Array of real elements of size
            (n,), where 'n' is the number of independent variables.
        ftol : float
            Tolerance for termination by the change of the cost function.
        xtol : float
            Tolerance for termination by the change of the independent variables.
        gtol : float
            Tolerance for termination by the norm of the gradient.
        max_nfev : int
            Maximum number of function evaluations.
        f_scale : float
            Cost function scalar
        x_scale : jnp.ndarray
            Scaling factors for independent variables.
        loss_function : callable, optional
            Loss function. If None, the standard least-squares problem is
            solved.
        tr_options : dict
            Options for the trust-region algorithm.
        verbose : int
            Level of algorithm's verbosity: 
                * 0 (default) : work silently.
                * 1 : display a termination report.

        Returns
        -------
        result : OptimizeResult
            The optimization result represented as a ``OptimizeResult`` object.
            Important attributes are: ``x`` the solution array, ``success`` a
            Boolean flag indicating if the optimizer exited successfully and
            ``message`` which describes the cause of the termination. See
            `OptimizeResult` for a description of other attributes.

        Notes
        -----
        The algorithm is described in [1]_.

        References
        ----------
        .. [1] J. J. More, "The Levenberg-Marquardt Algorithm: Implementation and
                Theory," Numerical Analysis, ed. G. A. Watson, Lecture Notes in
                Mathematics 630, Springer Verlag, pp. 105-116, 1977.

        """

        x = x0.copy()
        f_true = f
        nfev = 1
        njev = 1
        m, n = J.shape
    
        if loss_function is not None:
            rho = loss_function(f, f_scale)
            cost_jnp = self.calculate_cost(rho, data_mask)
            J, f = self.cJIT.scale_for_robust_loss_function(J, f, rho)
        else:      
            cost_jnp = self.default_loss_func(f)
        cost = np.array(cost_jnp)

        g_jnp = self.compute_grad(J, f)
        g = np.array(g_jnp)
        jac_scale = isinstance(x_scale, str) and x_scale == 'jac'
        if jac_scale:
            scale, scale_inv = self.cJIT.compute_jac_scale(J)
        else:
            scale, scale_inv = x_scale, 1 / x_scale
    
        Delta = norm(x0 * scale_inv)
        if Delta == 0:
            Delta = 1.0
    
        if max_nfev is None:
            max_nfev = x0.size * 100
    
        alpha = 0.0  # "Levenberg-Marquardt" parameter
    
        termination_status = None
        iteration = 0
        step_norm = None
        actual_reduction = None
    
        if verbose == 2:
            print_header_nonlinear()
    
        while True:
            g_norm = norm(g, ord=np.inf)
            if g_norm < gtol:
                termination_status = 1
                  
            if verbose == 2:
                print_iteration_nonlinear(iteration, nfev, cost, actual_reduction,
                                          step_norm, g_norm)
    
            if termination_status is not None or nfev == max_nfev:
                break
    
            d = scale
            d_jnp = jnp.array(scale)
            g_h_jnp = self.compute_grad_hat(g_jnp, d_jnp)
            svd_output = self.svd_no_bounds(J, d_jnp, f)

            J_h = svd_output[0]
            s, V, uf = [np.array(val) for val in svd_output[2:]]

            actual_reduction = -1
            while actual_reduction <= 0 and nfev < max_nfev:
                step_h, alpha, n_iter = solve_lsq_trust_region(
                    n, m, uf, s, V, Delta, initial_alpha=alpha)
    
                predicted_reduction_jnp = -self.cJIT.evaluate_quadratic(J_h, g_h_jnp, 
                                                                    step_h)
                predicted_reduction = np.array(predicted_reduction_jnp)

                step = d * step_h
                x_new = x + step
                
                f_new = fun(x_new, xdata, ydata, data_mask, transform)
                
                nfev += 1
    
                step_h_norm = norm(step_h)
    
                if not self.check_isfinite(f_new):
                    Delta = 0.25 * step_h_norm
                    continue
    
                if loss_function is not None:
                    cost_new_jnp = loss_function(f_new, f_scale, data_mask, 
                                                 cost_only=True)
                else:
                    cost_new_jnp = self.default_loss_func(f_new)
                cost_new = np.array(cost_new_jnp)

                actual_reduction = cost - cost_new
 
                Delta_new, ratio = update_tr_radius(
                    Delta, actual_reduction, predicted_reduction,
                    step_h_norm, step_h_norm > 0.95 * Delta)
    
                step_norm = norm(step)
                termination_status = check_termination(
                    actual_reduction, cost, step_norm, norm(x), ratio, ftol, xtol)

                if termination_status is not None:
                    break
    
                alpha *= Delta / Delta_new
                Delta = Delta_new
    
            if actual_reduction > 0:
                x = x_new
                f = f_new
                f_true = f
                cost = cost_new
                J = jac(x, xdata, ydata, data_mask, transform)
                njev += 1
                
                if loss_function is not None:
                    rho = loss_function(f, f_scale)
                    J, f = self.cJIT.scale_for_robust_loss_function(J, f, rho)
                    
                g_jnp = self.compute_grad(J, f)
                g = np.array(g_jnp)
                if jac_scale:
                    scale, scale_inv = self.cJIT.compute_jac_scale(J, scale_inv)
            else:
                step_norm = 0
                actual_reduction = 0
    
            iteration += 1
    
        if termination_status is None:
            termination_status = 0
            
        active_mask = np.zeros_like(x)
        return OptimizeResult(
            x=x, cost=cost, fun=f_true, jac=J, grad=g, optimality=g_norm,
            active_mask=active_mask, nfev=nfev, njev=njev,
            status=termination_status, all_times={})


    def trf_bounds(self, 
                   fun: Callable, 
                    xdata: Union[jnp.ndarray, Tuple[jnp.ndarray]], 
                   ydata: jnp.ndarray, 
                   jac: Callable, 
                   data_mask: jnp.ndarray, 
                   transform: jnp.ndarray, 
                   x0: np.ndarray, 
                   f: jnp.ndarray, 
                   J: jnp.ndarray, 
                   lb: np.ndarray, 
                   ub: np.ndarray,    
                   ftol: float, 
                   xtol: float, 
                   gtol: float, 
                   max_nfev: int,
                   f_scale: float, 
                   x_scale: np.ndarray, 
                   loss_function: Union[None, Callable], 
                   tr_options: Dict, 
                   verbose: int
                   ) -> Dict:

        """Bounded version of the trust-region reflective algorithm.

        Parameters
        ----------
        fun : callable
            The residual function
        xdata : array_like or tuple of array_like
            The independent variable where the data is measured. If `xdata` is a
            tuple, then the input arguments to `fun` are assumed to be
            ``(xdata[0], xdata[1], ...)``.
        ydata : jnp.ndarray
            The dependent data
        jac : callable
            The Jacobian of `fun`.
        data_mask : jnp.ndarray
            The mask for the data.
        transform : jnp.ndarray
            The uncertainty transform for the data.
        x0 : jnp.ndarray
            Initial guess. Array of real elements of size (n,), where 'n' is the
            number of independent variables.
        f0 : jnp.ndarray
            Initial residuals. Array of real elements of size (m,), where 'm' is
            the number of data points.
        J0 : jnp.ndarray
            Initial Jacobian. Array of real elements of size (m, n), where 'm' is
            the number of data points and 'n' is the number of independent
            variables.
        lb : jnp.ndarray
            Lower bounds on independent variables. Array of real elements of size
            (n,), where 'n' is the number of independent variables. 
        ub : jnp.ndarray
            Upper bounds on independent variables. Array of real elements of size
            (n,), where 'n' is the number of independent variables.
        ftol : float
            Tolerance for termination by the change of the cost function.
        xtol : float
            Tolerance for termination by the change of the independent variables.
        gtol : float
            Tolerance for termination by the norm of the gradient.
        max_nfev : int
            Maximum number of function evaluations.
        f_scale : float
            Cost function scalar
        x_scale : jnp.ndarray
            Scaling factors for independent variables.
        loss_function : callable, optional
            Loss function. If None, the standard least-squares problem is
            solved.
        tr_options : dict
            Options for the trust-region algorithm.
        verbose : int
            Level of algorithm's verbosity: 
                * 0 (default) : work silently.
                * 1 : display a termination report.

        Returns
        -------
        result : OptimizeResult
            The optimization result represented as a ``OptimizeResult`` object.
            Important attributes are: ``x`` the solution array, ``success`` a
            Boolean flag indicating if the optimizer exited successfully and
            ``message`` which describes the cause of the termination. See
            `OptimizeResult` for a description of other attributes.

        Notes
        -----
        The algorithm is described in [1]_.

        References
        ----------
        .. [1] J. J. More, "The Levenberg-Marquardt Algorithm: Implementation and
                Theory," in Numerical Analysis, ed. G. A. Watson (1978), pp. 105-116.
                DOI: 10.1017/CBO9780511819595.006
        .. [2] T. F. Coleman and Y. Li, “An interior trust region approach for 
                nonlinear minimization subject to bounds,” SIAM Journal on 
                Optimization, vol. 6, no. 2, pp. 418–445, 1996.
        """
                
        x = x0.copy()
        f_true = f
        nfev = 1
        njev = 1
        m, n = J.shape
        
        if loss_function is not None:
            rho = loss_function(f, f_scale)
            cost_jnp = self.calculate_cost(rho, data_mask)
            J, f = self.cJIT.scale_for_robust_loss_function(J, f, rho)
        else:
            cost_jnp = self.default_loss_func(f)
            
        cost = np.array(cost_jnp)

        g_jnp = self.compute_grad(J, f)
        g = np.array(g_jnp)
        
        jac_scale = isinstance(x_scale, str) and x_scale == 'jac'
        if jac_scale:
            scale, scale_inv = self.cJIT.compute_jac_scale(J)
        else:
            scale, scale_inv = x_scale, 1 / x_scale
        

        v, dv = CL_scaling_vector(x, g, lb, ub)
    
        v[dv != 0] *= scale_inv[dv != 0]
        Delta = norm(x0 * scale_inv / v**0.5)
        if Delta == 0:
            Delta = 1.0
    
        g_norm = norm(g * v, ord=np.inf)
        
        if max_nfev is None:
            max_nfev = x0.size * 100
    
        alpha = 0.0  # "Levenberg-Marquardt" parameter
    
        termination_status = None
        iteration = 0
        step_norm = None
        actual_reduction = None
    
        if verbose == 2:
            print_header_nonlinear()
        
        f_zeros = jnp.zeros([n])
        while True:

            v, dv = CL_scaling_vector(x, g, lb, ub)
    
            g_norm = norm(g * v, ord=np.inf)
            if g_norm < gtol:
                termination_status = 1
    
            # if verbose == 2:
            #     print_iteration_nonlinear(iteration, nfev, cost, actual_reduction,
            #                               step_norm, g_norm)
    
            if termination_status is not None or nfev == max_nfev:
                break
    
            # Now compute variables in "hat" space. Here, we also account for
            # scaling introduced by `x_scale` parameter. This part is a bit tricky,
            # you have to write down the formulas and see how the trust-region
            # problem is formulated when the two types of scaling are applied.
            # The idea is that first we apply `x_scale` and then apply Coleman-Li
            # approach in the new variables.
    
            # v is recomputed in the variables after applying `x_scale`, note that
            # components which were identically 1 not affected.
            v[dv != 0] *= scale_inv[dv != 0]
    
            # Here, we apply two types of scaling.
            d = v**0.5 * scale
    
            # C = diag(g * scale) Jv
            diag_h = g * dv * scale
    
            # After all this has been done, we continue normally.
    
            # "hat" gradient.
            g_h = d * g
            J_diag = jnp.diag(diag_h**0.5)
            d_jnp = jnp.array(d)

            output = self.svd_bounds(f, J, d_jnp, J_diag, f_zeros)
            J_h = output[0]
            s, V, uf = [np.array(val) for val in output[2:]]

 
            # theta controls step back step ratio from the bounds.
            theta = max(0.995, 1 - g_norm)
    
            actual_reduction = -1
            while actual_reduction <= 0 and nfev < max_nfev:
                p_h, alpha, n_iter = solve_lsq_trust_region(
                    n, m, uf, s, V, Delta, initial_alpha=alpha)
                
                p = d * p_h  # Trust-region solution in the original space.
                step, step_h, predicted_reduction = self.select_step(
                    x, J_h, diag_h, g_h, p, p_h, d, Delta, lb, ub, theta)
                
                x_new = make_strictly_feasible(x + step.copy(), lb, ub, rstep=0)
                f_new = fun(x_new, xdata, ydata, data_mask, transform)

                nfev += 1
    
                step_h_norm = norm(step_h)
                if not self.check_isfinite(f_new):
                    Delta = 0.25 * step_h_norm
                    continue

                if loss_function is not None:
                    cost_new_jnp = loss_function(f_new, f_scale, data_mask, 
                                                 cost_only=True)
                else:
                    cost_new_jnp = self.default_loss_func(f_new)
                cost_new = np.array(cost_new_jnp)

                actual_reduction = cost - cost_new
                Delta_new, ratio = update_tr_radius(
                    Delta, actual_reduction, predicted_reduction,
                    step_h_norm, step_h_norm > 0.95 * Delta)

    
                step_norm = norm(step)
                termination_status = check_termination(
                    actual_reduction, cost, step_norm, norm(x), ratio, ftol, xtol)
                if termination_status is not None:
                    break
    
                alpha *= Delta / Delta_new
                Delta = Delta_new



            if actual_reduction > 0:
                x = x_new
                f = f_new
                f_true = f

                cost = cost_new
    
                J = jac(x, xdata, ydata, data_mask, transform)

                njev += 1
                
                if loss_function is not None:
                    rho = loss_function(f, f_scale)
                    J, f = self.cJIT.scale_for_robust_loss_function(J, f, rho)

                g_jnp = self.compute_grad(J, f)
                if jac_scale:
                    scale, scale_inv = self.cJIT.compute_jac_scale(J, scale_inv)
                g = np.array(g_jnp)
            else:
                step_norm = 0
                actual_reduction = 0
    
            iteration += 1
    
        if termination_status is None:
            termination_status = 0
                
        active_mask = find_active_constraints(x, lb, ub, rtol=xtol)
        return OptimizeResult(
            x=x, cost=cost, fun=f_true, jac=J, grad=g, optimality=g_norm,
            active_mask=active_mask, nfev=nfev, njev=njev,
            status=termination_status)
    
    def select_step(self, 
                    x: np.ndarray, 
                    J_h: jnp.ndarray, 
                    diag_h: jnp.ndarray, 
                    g_h: jnp.ndarray, 
                    p: np.ndarray, 
                    p_h: np.ndarray, 
                    d: np.ndarray, 
                    Delta: float, 
                    lb: np.ndarray, 
                    ub: np.ndarray, 
                    theta: float
                    ):
        """Select the best step according to Trust Region Reflective algorithm.
        
        Parameters
        ----------
        x : np.ndarray
            Current set parameter vector.
        J_h : jnp.ndarray
            Jacobian matrix in the scaled 'hat' space.
        diag_h : jnp.ndarray
            Diagonal of the scaled matrix C = diag(g * scale) Jv?
        g_h : jnp.ndarray
            Gradient vector in the scaled 'hat' space.
        p : np.ndarray
            Trust-region step in the original space.
        p_h : np.ndarray
            Trust-region step in the scaled 'hat' space.
        d : np.ndarray
            Scaling vector.
        Delta : float
            Trust-region radius.
        lb : np.ndarray
            Lower bounds on variables.
        ub : np.ndarray
            Upper bounds on variables.
        theta : float
            Controls step back step ratio from the bounds.

        Returns
        -------
        step : np.ndarray   
            Step in the original space.
        step_h : np.ndarray
            Step in the scaled 'hat' space.
        predicted_reduction : float
            Predicted reduction in the cost function.        
        """
        if in_bounds(x + p, lb, ub):
            p_value = self.cJIT.evaluate_quadratic(J_h, g_h, p_h, diag=diag_h)
            return p, p_h, -p_value
    
        p_stride, hits = step_size_to_bound(x, p, lb, ub)
    
        # Compute the reflected direction.
        r_h = np.copy(p_h)
        r_h[hits.astype(bool)] *= -1
        r = d * r_h
    
        # Restrict trust-region step, such that it hits the bound.
        p *= p_stride
        p_h *= p_stride
        x_on_bound = x + p
    
        # Reflected direction will cross first either feasible region or trust
        # region boundary.
        _, to_tr = intersect_trust_region(p_h, r_h, Delta)
        to_bound, _ = step_size_to_bound(x_on_bound, r, lb, ub)
    
        # Find lower and upper bounds on a step size along the reflected
        # direction, considering the strict feasibility requirement. There is no
        # single correct way to do that, the chosen approach seems to work best
        # on test problems.
        r_stride = min(to_bound, to_tr)
        if r_stride > 0:
            r_stride_l = (1 - theta) * p_stride / r_stride
            if r_stride == to_bound:
                r_stride_u = theta * to_bound
            else:
                r_stride_u = to_tr
        else:
            r_stride_l = 0
            r_stride_u = -1
    
        # Check if reflection step is available.
        if r_stride_l <= r_stride_u:
            a, b, c = self.cJIT.build_quadratic_1d(J_h, g_h, r_h, s0=p_h,
                                                   diag=diag_h)

            r_stride, r_value = minimize_quadratic_1d(
                a, b, r_stride_l, r_stride_u, c=c)
            r_h *= r_stride
            r_h += p_h
            r = r_h * d
        else:
            r_value = np.inf
    
        # Now correct p_h to make it strictly interior.
        p *= theta
        p_h *= theta
        p_value = self.cJIT.evaluate_quadratic(J_h, g_h, p_h, diag=diag_h)
    
        ag_h = -g_h
        ag = d * ag_h
    
        to_tr = Delta / norm(ag_h)
        to_bound, _ = step_size_to_bound(x, ag, lb, ub)
        if to_bound < to_tr:
            ag_stride = theta * to_bound
        else:
            ag_stride = to_tr
    
        a, b = self.cJIT.build_quadratic_1d(J_h, g_h, ag_h, diag=diag_h)
        ag_stride, ag_value = minimize_quadratic_1d(a, b, 0, ag_stride)
        ag_h *= ag_stride
        ag *= ag_stride
    
        if p_value < r_value and p_value < ag_value:
            return p, p_h, -p_value
        elif r_value < p_value and r_value < ag_value:
            return r, r_h, -r_value
        else:
            return ag, ag_h, -ag_value
        
        
    def trf_no_bounds_timed(self, 
                      fun: Callable, 
                      xdata: Union[jnp.ndarray, Tuple[jnp.ndarray]], 
                      ydata: jnp.ndarray, 
                      jac: Callable, 
                      data_mask: jnp.ndarray, 
                      transform: jnp.ndarray, 
                      x0: np.ndarray, 
                      f: jnp.ndarray, 
                      J: jnp.ndarray, 
                      lb: np.ndarray, 
                      ub: np.ndarray,    
                      ftol: float, 
                      xtol: float, 
                      gtol: float, 
                      max_nfev: int,
                      f_scale: float, 
                      x_scale: np.ndarray, 
                      loss_function: Union[None, Callable], 
                      tr_options: Dict, 
                      verbose: int
                      ) -> Dict:
        """Trust Region Reflective algorithm with no bounds and all the 
        operations performed on JAX and the GPU are timed. We need a separate
        function for this because to time each operation we need a 
        block_until_ready() function which makes the main Python thread wait
        until the GPU has finished the operation. However, for the main
        algorithm we don't want to wait for the GPU to finish each operation
        because it would slow down the algorithm. Thus, this is just used for
        analysis of the algorithm.

        Parameters
        ----------
        fun : callable
            The residual function
        xdata : array_like or tuple of array_like
            The independent variable where the data is measured. If `xdata` is a
            tuple, then the input arguments to `fun` are assumed to be
            ``(xdata[0], xdata[1], ...)``.
        ydata : jnp.ndarray
            The dependent data
        jac : callable
            The Jacobian of `fun`.
        data_mask : jnp.ndarray
            The mask for the data.
        transform : jnp.ndarray
            The uncertainty transform for the data.
        x0 : jnp.ndarray
            Initial guess. Array of real elements of size (n,), where 'n' is the
            number of independent variables.
        f0 : jnp.ndarray
            Initial residuals. Array of real elements of size (m,), where 'm' is
            the number of data points.
        J0 : jnp.ndarray
            Initial Jacobian. Array of real elements of size (m, n), where 'm' is
            the number of data points and 'n' is the number of independent
            variables.
        lb : jnp.ndarray
            Lower bounds on independent variables. Array of real elements of size
            (n,), where 'n' is the number of independent variables. 
        ub : jnp.ndarray
            Upper bounds on independent variables. Array of real elements of size
            (n,), where 'n' is the number of independent variables.
        ftol : float
            Tolerance for termination by the change of the cost function.
        xtol : float
            Tolerance for termination by the change of the independent variables.
        gtol : float
            Tolerance for termination by the norm of the gradient.
        max_nfev : int
            Maximum number of function evaluations.
        f_scale : float
            Cost function scalar
        x_scale : jnp.ndarray
            Scaling factors for independent variables.
        loss_function : callable, optional
            Loss function. If None, the standard least-squares problem is
            solved.
        tr_options : dict
            Options for the trust-region algorithm.
        verbose : int
            Level of algorithm's verbosity: 
                * 0 (default) : work silently.
                * 1 : display a termination report.

        Returns
        -------
        result : OptimizeResult
            The optimization result represented as a ``OptimizeResult`` object.
            Important attributes are: ``x`` the solution array, ``success`` a
            Boolean flag indicating if the optimizer exited successfully and
            ``message`` which describes the cause of the termination. See
            `OptimizeResult` for a description of other attributes.

        Notes
        -----
        The algorithm is described in [1]_.

        References
        ----------
        .. [1] J. J. More, "The Levenberg-Marquardt Algorithm: Implementation and
                Theory," Numerical Analysis, ed. G. A. Watson, Lecture Notes in
                Mathematics 630, Springer Verlag, pp. 105-116, 1977.
        """
      
        ftimes = []
        jtimes = []
        svd_times = []
        ctimes = []
        gtimes = []
        gtimes2 = []
        ptimes = []

        svd_ctimes = []
        g_ctimes = []
        c_ctimes = []
        p_ctimes = []
     
        x = x0.copy()
        
        st = time.time()
        f = fun(x, xdata, ydata, data_mask, transform).block_until_ready()
        ftimes.append(time.time() - st)
        f_true = f
        nfev = 1
        
        st = time.time()
        J = jac(x, xdata, ydata, data_mask, transform).block_until_ready()
        jtimes.append(time.time() - st)

        
        njev = 1
        m, n = J.shape
    
        if loss_function is not None:
            rho = loss_function(f, f_scale)
            cost_jnp = self.calculate_cost(rho, data_mask)
            J, f = self.cJIT.scale_for_robust_loss_function(J, f, rho)
        else:      
            st1 = time.time()
            cost_jnp = self.default_loss_func(f).block_until_ready()
            st2 = time.time()
        cost = np.array(cost_jnp)
        st3 = time.time()

        ctimes.append(st2 - st1)
        c_ctimes.append(st3 - st2)
        
        st1 = time.time()
        g_jnp = self.compute_grad(J, f).block_until_ready()
        st2 = time.time()
        g = np.array(g_jnp)
        st3 = time.time()


        gtimes.append(st2 - st1)
        g_ctimes.append(st3 - st2)
        
        jac_scale = isinstance(x_scale, str) and x_scale == 'jac'
        if jac_scale:
            scale, scale_inv = self.cJIT.compute_jac_scale(J)
        else:
            scale, scale_inv = x_scale, 1 / x_scale
    
        Delta = norm(x0 * scale_inv)
        if Delta == 0:
            Delta = 1.0
    
        if max_nfev is None:
            max_nfev = x0.size * 100
    
        alpha = 0.0  # "Levenberg-Marquardt" parameter
    
        termination_status = None
        iteration = 0
        step_norm = None
        actual_reduction = None
    
        if verbose == 2:
            print_header_nonlinear()
    
        while True:
            g_norm = norm(g, ord=np.inf)
            if g_norm < gtol:
                termination_status = 1
                  
            # if verbose == 2:
            #     print_iteration_nonlinear(iteration, nfev, cost, actual_reduction,
            #                               step_norm, g_norm)
    
            if termination_status is not None or nfev == max_nfev:
                break
    
            d = scale
            d_jnp = jnp.array(scale)

            # g_h = d * g
            g_h_jnp = self.compute_grad_hat(g_jnp, d_jnp)
            
            st = time.time()
            svd_output = self.svd_no_bounds(J, d_jnp, f)
            tree_flatten(svd_output)[0][0].block_until_ready()
            svd_times.append(time.time() - st)
            J_h = svd_output[0]
            
            st = time.time()
            s, V, uf = [np.array(val) for val in svd_output[2:]]
            svd_ctimes.append(time.time() - st)

            
            actual_reduction = -1
            while actual_reduction <= 0 and nfev < max_nfev:
                step_h, alpha, n_iter = solve_lsq_trust_region(
                    n, m, uf, s, V, Delta, initial_alpha=alpha)
    
                st1 = time.time()
                predicted_reduction_jnp = -self.cJIT.evaluate_quadratic(J_h, g_h_jnp, 
                                                                    step_h).block_until_ready()
                st2 = time.time()
                predicted_reduction = np.array(predicted_reduction_jnp)
                st3 = time.time()
                ptimes.append(st2 - st1)
                p_ctimes.append(st3 - st2)

                step = d * step_h
                x_new = x + step
                
                st = time.time()
                f_new = fun(x_new, xdata, ydata, data_mask, transform).block_until_ready()
                ftimes.append(time.time() - st)
                
                nfev += 1
    
                step_h_norm = norm(step_h)
    
                if not self.check_isfinite(f_new):
                    Delta = 0.25 * step_h_norm
                    continue
    
                if loss_function is not None:
                    cost_new_jnp = loss_function(f_new, f_scale, data_mask, 
                                                 cost_only=True)
                else:
                    st1 = time.time()
                    cost_new_jnp = self.default_loss_func(f_new).block_until_ready()
                    st2 = time.time()
                    cost_new = np.array(cost_new_jnp)
                    st3 = time.time()

                    ctimes.append(st2 - st1)
                    c_ctimes.append(st3 - st2)

                actual_reduction = cost - cost_new
 
                Delta_new, ratio = update_tr_radius(
                    Delta, actual_reduction, predicted_reduction,
                    step_h_norm, step_h_norm > 0.95 * Delta)
    
                step_norm = norm(step)
                termination_status = check_termination(
                    actual_reduction, cost, step_norm, norm(x), ratio, ftol, xtol)

                if termination_status is not None:
                    break
    
                alpha *= Delta / Delta_new
                Delta = Delta_new
    
            if actual_reduction > 0:
                x = x_new
    
                f = f_new
                f_true = f
    
                cost = cost_new
                
                st = time.time()
                J = jac(x, xdata, ydata, data_mask, transform).block_until_ready()
                jtimes.append(time.time() - st)
                
                njev += 1
                
                if loss_function is not None:
                    rho = loss_function(f, f_scale)
                    J, f = self.cJIT.scale_for_robust_loss_function(J, f, rho)
                    
                    
                st1 = time.time()
                g_jnp = self.compute_grad(J, f).block_until_ready()
                st2 = time.time()
                g = np.array(g_jnp)
                st3 = time.time()

                gtimes.append(st2 - st1)
                g_ctimes.append(st3 - st2)
    
                if jac_scale:
                    scale, scale_inv = self.cJIT.compute_jac_scale(J, scale_inv)
                
            else:
                step_norm = 0
                actual_reduction = 0
    
            iteration += 1
    
        if termination_status is None:
            termination_status = 0
            
        active_mask = np.zeros_like(x)

        tlabels = ['ftimes', 'jtimes', 'svd_times', 'ctimes', 'gtimes', 'ptimes', 
                   'g_ctimes', 'c_ctimes', 'svd_ctimes', 'p_ctimes', 'gtimes2']
        all_times = [ftimes, jtimes, svd_times, ctimes, gtimes, ptimes, 
                     g_ctimes, c_ctimes, svd_ctimes, p_ctimes, gtimes2]
                     
        tdicts = dict(zip(tlabels, all_times))
        return OptimizeResult(
            x=x, cost=cost, fun=f_true, jac=J, grad=g, optimality=g_norm,
            active_mask=active_mask, nfev=nfev, njev=njev,
            status=termination_status, all_times=tdicts)