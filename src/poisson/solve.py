from __future__ import annotations
from typing import Optional, Dict, Any
from dolfinx.fem.petsc import LinearProblem

def solve_linear(a, L, bcs, mpc=None, solver_cfg: Optional[Dict[str, Any]] = None):
    opts = {"ksp_type":"cg","pc_type":"hypre","ksp_rtol":1e-10,"ksp_error_if_not_converged":True}
    if solver_cfg: opts.update(solver_cfg)
    if mpc is None:
        problem = LinearProblem(a, L, bcs=bcs, petsc_options=opts, petsc_options_prefix="lp_")
        uh = problem.solve()
        info = {"its": problem.solver.getIterationNumber()}
        return uh, info
    else:
        # Wire MPC later (dolfinx_mpc.LinearProblem)
        raise NotImplementedError("MPC solve not yet implemented.")
