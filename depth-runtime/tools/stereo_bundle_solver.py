"""Bounded offline exact TRF comparison for small stereo bundle problems.

The default remains sparse LSMR. The opt-in exact step uses the same colored
finite differences, bounds, robust loss and termination tolerances. SciPy's
finite-difference helper is private; its installed version is recorded and its
Jacobian is checked against independent analytic controls. No live dependency.
"""
import numpy as np
import hashlib
from pathlib import Path
from scipy.sparse import csc_matrix

MAX_DENSE_JACOBIAN_ENTRIES = 2_000_000


def colored_dense_jacobian(fun, x, sparsity, bounds):
    from scipy.optimize._numdiff import approx_derivative
    if sparsity.shape[0]*sparsity.shape[1] > MAX_DENSE_JACOBIAN_ENTRIES:
        raise ValueError('dense Jacobian exceeds explicit entry budget')
    return approx_derivative(fun,x,method='2-point',sparsity=sparsity,bounds=bounds).toarray()


def solve_bundle(fun, start, sparsity, bounds, loss, f_scale, max_nfev, exact_small_problem=False,nuisance_blocks=None):
    import scipy
    from scipy.optimize import least_squares
    if not isinstance(exact_small_problem,bool):
        raise ValueError('exact_small_problem must be boolean')
    entries=int(sparsity.shape[0])*int(sparsity.shape[1])
    exact=exact_small_problem and entries <= MAX_DENSE_JACOBIAN_ENTRIES
    options=dict(bounds=bounds,loss=loss,f_scale=f_scale,x_scale='jac',
                 ftol=1e-7,xtol=1e-7,gtol=1e-7,max_nfev=int(max_nfev))
    transform=None;preconditioner=None;working_optimality=None
    if nuisance_blocks is not None:
        from scipy.optimize._numdiff import approx_derivative
        from stereo_block_preconditioner import block_rotation
        initial_jac=approx_derivative(fun,start,method='2-point',sparsity=sparsity,bounds=bounds)
        transform,preconditioner=block_rotation(initial_jac,nuisance_blocks,bounds)
        def decoded(q):return np.asarray(transform@q).ravel()
        def jacobian(q):
            physical=approx_derivative(fun,decoded(q),method='2-point',sparsity=sparsity,bounds=bounds)
            working=physical@transform
            return working.toarray() if exact else working.tocsr()
        options.update(jac=jacobian,tr_solver='exact' if exact else 'lsmr')
        if not exact:options['tr_options']={'atol':1e-10,'btol':1e-10,'maxiter':1000}
        result=least_squares(lambda q:fun(decoded(q)),np.asarray(transform.T@start).ravel(),**options)
        working_optimality=float(result.optimality)
        result.x=decoded(result.x)
        result.jac=csc_matrix(result.jac)@transform.T
        result.grad=np.asarray(transform@result.grad).ravel()
        lower=np.broadcast_to(bounds[0],result.x.shape);upper=np.broadcast_to(bounds[1],result.x.shape)
        scale=np.ones_like(result.x)
        use=(result.grad<0)&np.isfinite(upper);scale[use]=upper[use]-result.x[use]
        use=(result.grad>0)&np.isfinite(lower);scale[use]=result.x[use]-lower[use]
        result.optimality=float(np.max(np.abs(result.grad*scale),initial=0.))
        if result.status==1 and result.optimality>options['gtol']:
            result.success=False;result.status=0
            result.message='Working gradient converged, but original-coordinate gradient did not.'
    else:
        if exact:
            options.update(jac=lambda x:colored_dense_jacobian(fun,x,sparsity,bounds),tr_solver='exact')
        else:
            options.update(jac_sparsity=sparsity,tr_options={'atol':1e-10,'btol':1e-10,'maxiter':1000})
        result=least_squares(fun,start,**options)
        if exact:result.jac=csc_matrix(result.jac)
    return result,{'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'solver':'dense_exact' if exact else 'sparse_lsmr',
        'exact_small_problem_requested':exact_small_problem,'jacobian_entries':entries,
        'dense_entry_limit':MAX_DENSE_JACOBIAN_ENTRIES,'scipy_version':scipy.__version__,'block_preconditioner':preconditioner,'working_optimality':working_optimality}
