"""Sparse nuisance projection with unit-column preconditioning.

Scaling nuisance columns preserves their span and therefore the exact profiled
calibration Jacobian. Calibration columns themselves and tolerances are unchanged.
"""
from pathlib import Path
import hashlib
import numpy as np
from scipy.sparse.linalg import lsmr


def profile_calibration_columns(jacobian,calibration_columns,max_iterations=2000,recovery_iterations=20000):
    # Retry only iteration exhaustion. Tolerances, column span and acceptance
    # stop codes stay unchanged; this does not change the fitted objective.
    for budget in (max_iterations,recovery_iterations):
        if isinstance(budget,bool) or not isinstance(budget,int) or budget<1:
            raise ValueError("positive integer projection iteration budgets required")
    jac=jacobian.tocsc();nuisance=jac[:,calibration_columns:]
    norms=np.sqrt(np.asarray(nuisance.multiply(nuisance).sum(axis=0)).ravel())
    inverse=np.divide(1.,norms,out=np.ones_like(norms),where=norms>0)
    scaled=nuisance.multiply(inverse).tocsc()
    columns=[];stops=[];iterations=[];residuals=[];conditions=[];attempts=[]
    for j in range(calibration_columns):
        v=jac[:,j].toarray().ravel()
        solution=lsmr(scaled,v,atol=1e-8,btol=1e-8,maxiter=max_iterations)
        history=[{'budget':max_iterations,'stop':int(solution[1]),'iterations':int(solution[2])}]
        if solution[1]==7 and recovery_iterations>max_iterations:
            solution=lsmr(scaled,v,atol=1e-8,btol=1e-8,maxiter=recovery_iterations)
            history.append({'budget':recovery_iterations,'stop':int(solution[1]),'iterations':int(solution[2])})
        attempts.append(history)
        columns.append(v-scaled@solution[0]);stops.append(int(solution[1]));iterations.append(int(solution[2]));residuals.append(float(solution[3]));conditions.append(float(solution[6]))
    return np.column_stack(columns),{'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'nuisance_column_scaling':'unit_l2_preserves_column_span','zero_nuisance_columns':int(np.sum(norms==0)),
        'attempts':attempts,'initial_iteration_budget':max_iterations,'recovery_iteration_budget':recovery_iterations,
        'stops':stops,'iterations':iterations,'residual_norms':residuals,'scaled_condition_estimates':conditions}
