"""MILP solver for the polling place location problem (gurobipy).

Capacitated facility location with assignment:

    min    sum_{i,j}  v_i * d_ij * y_ij
    s.t.   sum_j y_ij = 1                       for all i      (each precinct assigned once)
           y_ij <= x_j                           for all i, j   (assign only to opened sites)
           sum_i v_i * y_ij <= cap_j * x_j      for all j      (capacity)
           sum_j x_j <= K                                       (budget)
           x, y in {0, 1}

Agent actions are encoded as variable fixings on the same MIP:
    force_open[j]      :  fix x_j = 1
    force_close[j]     :  fix x_j = 0
    force_assign[i,j]  :  fix y_ij = 1   (also pins x_j = 1 via the link constraint)

Requires `gurobipy`. A free academic licence is available from gurobi.com.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from instance import Instance, Solution


@dataclass
class FixingConstraints:
    """Modifications to the MILP corresponding to the agent's actions.

    Hard variable fixings (force the optimizer's hand):
        force_open[j]      : x_j = 1
        force_close[j]     : x_j = 0
        force_assign[i, j] : y_ij = 1 (also implies x_j = 1)

    Soft objective modification (nudge without forcing):
        precinct_weight_multipliers : map of precinct_index -> multiplier (>=0).
            Default per-precinct weight is the precinct's voter count v_i.
            Effective objective becomes
                sum_{i,j} (m_i * v_i) * d_ij * y_ij
            where m_i defaults to 1.0 and the agent can boost it (e.g. 2.0)
            to prioritize a precinct, or reduce it (e.g. 0.5) to deprioritize.
    """
    force_open: List[int] = field(default_factory=list)
    force_close: List[int] = field(default_factory=list)
    force_assign: List[Tuple[int, int]] = field(default_factory=list)
    precinct_weight_multipliers: Dict[int, float] = field(default_factory=dict)


def solve_baseline(
    instance: Instance,
    constraints: Optional[FixingConstraints] = None,
    time_limit: float = 60.0,
    mip_gap: float = 0.005,
    threads: int = 0,
    verbose: bool = False,
    seed: int = 0,  # accepted for API compatibility; Gurobi uses its own seed param
) -> Solution:
    """Solve the polling place MILP with optional variable fixings.

    Parameters
    ----------
    instance : the problem instance.
    constraints : optional FixingConstraints encoding agent actions.
    time_limit : wall-clock seconds before stopping (returns best feasible).
    mip_gap : relative MIP gap stopping criterion (e.g. 0.005 = 0.5%).
    threads : 0 = use Gurobi's default thread count.
    verbose : if True, print Gurobi solver log.
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as e:
        raise ImportError(
            "gurobipy is required. Install with `pip install gurobipy`. "
            "A free academic licence is available from https://www.gurobi.com/."
        ) from e

    if constraints is None:
        constraints = FixingConstraints()

    # Validate fixings
    fc_set = set(constraints.force_close)
    fo_set = set(constraints.force_open)
    if fo_set & fc_set:
        raise ValueError("A site cannot be both force_open and force_close.")
    if any(j in fc_set for _, j in constraints.force_assign):
        raise ValueError("force_assign cannot reference a force_close site.")

    M = instance.n_sites
    N = instance.n_precincts
    K = instance.K
    voters = instance.precinct_voters
    capacity = instance.site_capacity
    D = instance.distance_matrix

    env = gp.Env(empty=True)
    if not verbose:
        env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model("polling_place", env=env)
    m.Params.TimeLimit = time_limit
    m.Params.MIPGap = mip_gap
    if threads > 0:
        m.Params.Threads = threads
    if seed != 0:
        m.Params.Seed = seed

    # Decision variables
    x = m.addVars(M, vtype=GRB.BINARY, name="x")
    y = m.addVars(N, M, vtype=GRB.BINARY, name="y")

    # Effective per-precinct weights = voters * agent-supplied multiplier.
    # Default multiplier is 1.0 for every precinct; the agent can boost a
    # precinct (m_i > 1) to prioritize its travel distance, or reduce it
    # (m_i < 1) to deprioritize. Multipliers are clamped to a sane range.
    MULT_MIN, MULT_MAX = 0.0, 100.0
    effective_weights = voters.astype(float).copy()
    for i, mult in (constraints.precinct_weight_multipliers or {}).items():
        idx = int(i)
        if 0 <= idx < N:
            mult_clamped = max(MULT_MIN, min(MULT_MAX, float(mult)))
            effective_weights[idx] *= mult_clamped

    # Objective: voter-weighted distance with per-precinct weight multipliers
    m.setObjective(
        gp.quicksum(float(effective_weights[i]) * float(D[i, j]) * y[i, j]
                    for i in range(N) for j in range(M)),
        GRB.MINIMIZE,
    )

    # Each precinct assigned to exactly one site
    for i in range(N):
        m.addConstr(gp.quicksum(y[i, j] for j in range(M)) == 1, name=f"assign_{i}")

    # Linking: assignment requires open site
    for i in range(N):
        for j in range(M):
            m.addConstr(y[i, j] <= x[j], name=f"link_{i}_{j}")

    # Capacity (also enforces y_ij = 0 if x_j = 0 via the right-hand side)
    for j in range(M):
        m.addConstr(
            gp.quicksum(float(voters[i]) * y[i, j] for i in range(N))
            <= float(capacity[j]) * x[j],
            name=f"cap_{j}",
        )

    # Budget on opened sites
    m.addConstr(gp.quicksum(x[j] for j in range(M)) <= K, name="budget")

    # Variable fixings (the agent's actions)
    for j in fo_set:
        x[j].LB = 1
        x[j].UB = 1
    for j in fc_set:
        x[j].LB = 0
        x[j].UB = 0
    for i, j in constraints.force_assign:
        y[i, j].LB = 1
        y[i, j].UB = 1

    m.optimize()
    status = int(m.Status)

    # Handle infeasibility / no-solution
    if status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        return Solution(
            x=np.zeros(M, dtype=np.int8),
            y=np.zeros((N, M), dtype=np.int8),
            objective=float("inf"),
            solver_status="infeasible",
            metadata={
                "feasible": False,
                "gurobi_status": status,
                "force_open": list(constraints.force_open),
                "force_close": list(constraints.force_close),
                "force_assign": list(constraints.force_assign),
            },
        )
    if m.SolCount == 0:
        return Solution(
            x=np.zeros(M, dtype=np.int8),
            y=np.zeros((N, M), dtype=np.int8),
            objective=float("inf"),
            solver_status=f"no-solution-status-{status}",
            metadata={"feasible": False, "gurobi_status": status},
        )

    # Extract solution
    x_val = np.array([x[j].X for j in range(M)])
    y_val = np.array([[y[i, j].X for j in range(M)] for i in range(N)])
    x_val = (x_val > 0.5).astype(np.int8)
    y_val = (y_val > 0.5).astype(np.int8)

    return Solution(
        x=x_val,
        y=y_val,
        objective=float(m.ObjVal),
        solver_status=f"gurobi-status-{status}",
        metadata={
            "feasible": True,
            "mip_gap": float(m.MIPGap),
            "runtime_sec": float(m.Runtime),
            "gurobi_status": status,
            "force_open": list(constraints.force_open),
            "force_close": list(constraints.force_close),
            "force_assign": list(constraints.force_assign),
            "precinct_weight_multipliers":
                dict(constraints.precinct_weight_multipliers or {}),
        },
    )
