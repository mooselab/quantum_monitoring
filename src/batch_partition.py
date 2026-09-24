"""
Batched-instrumentation partitioning for RQ3.

Partitioning the individually feasible monitor groups into instrumented
variants, each within the fixed per-run qubit budget and executed separately,
lets QMon read every feasible group; detection is the union over batches.

The problem is bin packing: minimize k subject to
    sum_{n in batch} extra_qubits(n) <= Qmax - num_original_qubits.
First-fit decreasing supplies the deployed packing.  For small instances, one
feasibility ILP tests whether one fewer batch suffices; this is a one-step
refinement, not a global-optimality claim.

PuLP is the model-construction layer. ``QMON_MILP_SOLVER`` selects the actual
backend; the final runs use Gurobi and CBC remains an explicit fallback.
"""
import hashlib
import importlib.metadata
import os
import time
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import pulp

SOLVER_BACKEND_ENV = "QMON_MILP_SOLVER"
DEFAULT_REFINEMENT_TIME_LIMIT_SECONDS = 600
DEFAULT_ILP_NODE_LIMIT = 120
GUROBI_LICENSE_RETRY_DELAYS_SECONDS = (15, 30, 60, 120, 240, 480)


def _is_transient_gurobi_session_error(error):
    """Recognize only Gurobi license/session failures that can clear on retry."""

    error_type = type(error)
    if error_type.__name__ != "GurobiError" or not error_type.__module__.startswith(
        "gurobipy"
    ):
        return False
    message = str(error).lower()
    markers = (
        "too many sessions",
        "active sessions",
        "web license service",
        "wls",
        "license server",
        "token server",
        "failed to acquire a license",
        "unable to acquire a license",
        "license checkout",
        "token.gurobi.com",
    )
    return any(marker in message for marker in markers)


def _close_managed_solver(solver):
    if solver is None or not getattr(solver, "manage_env", False):
        return
    failures = []
    model = getattr(solver, "model", None)
    environment = getattr(solver, "env", None)
    try:
        if model is not None:
            try:
                model.dispose()
            except Exception as exc:
                failures.append(("model", exc))
        if environment is not None:
            try:
                environment.dispose()
            except Exception as exc:
                failures.append(("environment", exc))
    finally:
        # PuLP's destructor calls close() again when this flag remains true.
        # Clear it even if either partial Gurobi object rejects disposal.
        solver.init_gurobi = False
    if failures:
        labels = ", ".join(label for label, _ in failures)
        primary = failures[0][1]
        raise RuntimeError(
            f"managed Gurobi cleanup failed for {labels}"
        ) from primary


def requested_backend():
    backend = os.environ.get(SOLVER_BACKEND_ENV, "gurobi").strip().lower()
    if backend not in {"gurobi", "cbc"}:
        raise ValueError(
            f"unsupported QMON_MILP_SOLVER={backend!r}; expected 'gurobi' or 'cbc'"
        )
    return backend


def make_milp_solver(*, time_limit=None):
    """Build the explicitly requested backend; never fall back silently."""

    backend = requested_backend()
    if backend == "gurobi":
        return pulp.GUROBI(
            msg=False,
            timeLimit=time_limit,
            manageEnv=True,
            MIPGap=0.0,
            Threads=1,
            Seed=0,
        )
    if backend == "cbc":
        return pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit, threads=1)


def _file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_gurobi_retry_delays(value=None):
    delays = (
        GUROBI_LICENSE_RETRY_DELAYS_SECONDS
        if value is None
        else tuple(value)
    )
    if any(type(delay) is not int or delay < 0 for delay in delays):
        raise ValueError(
            "Gurobi license retry delays must be non-negative integers"
        )
    return tuple(delays)


@lru_cache(maxsize=8)
def _solver_provenance_for_backend(backend, gurobi_retry_delays):
    provenance = {
        "modeling_api": "pulp",
        "pulp_version": importlib.metadata.version("pulp"),
        "backend": backend,
        "threads": 1,
        "seed": 0 if backend == "gurobi" else None,
        "relative_mip_gap": 0.0 if backend == "gurobi" else None,
        "transient_license_retry_delays_seconds": (
            list(gurobi_retry_delays)
            if backend == "gurobi"
            else []
        ),
    }
    if backend == "gurobi":
        provenance["backend_package"] = "gurobipy"
        provenance["backend_version"] = importlib.metadata.version("gurobipy")
    else:
        solver = pulp.PULP_CBC_CMD(msg=False, threads=1)
        binary = Path(solver.path).resolve()
        provenance["backend_package"] = "cbc executable"
        provenance["backend_version"] = None
        provenance["backend_binary"] = str(binary)
        provenance["backend_binary_sha256"] = _file_sha256(binary)
    return provenance


def solver_provenance(*, gurobi_license_retry_delays_seconds=None):
    """Describe the requested backend, including the CBC binary when used."""

    backend = requested_backend()
    retry_delays = _normalized_gurobi_retry_delays(
        gurobi_license_retry_delays_seconds
    )
    return deepcopy(_solver_provenance_for_backend(backend, retry_delays))


def solve_problem(
        problem, *, time_limit=None, require_optimal=False,
        gurobi_license_retry_delays_seconds=None):
    configured_retry_delays = _normalized_gurobi_retry_delays(
        gurobi_license_retry_delays_seconds
    )
    retry_delays = (
        configured_retry_delays
        if requested_backend() == "gurobi"
        else ()
    )
    for attempt in range(len(retry_delays) + 1):
        solver = None
        retry_delay = None
        try:
            solver = make_milp_solver(time_limit=time_limit)
            status = problem.solve(solver)
            break
        except Exception as error:
            if (
                not _is_transient_gurobi_session_error(error)
                or attempt == len(retry_delays)
            ):
                raise
            # PuLP may attach a partially initialized model to the problem.
            # Clear it before the next managed-environment attempt.
            if hasattr(problem, "solverModel"):
                problem.solverModel = None
            retry_delay = retry_delays[attempt]
        finally:
            # A managed Gurobi environment must be closed after every model.
            # Failed WLS handshakes also need partial cleanup so PuLP's
            # destructor never dereferences an uninitialized model.
            _close_managed_solver(solver)
        if retry_delay is not None:
            time.sleep(retry_delay)
    if require_optimal and status != pulp.LpStatusOptimal:
        raise RuntimeError(
            f"MILP returned {pulp.LpStatus.get(status, status)!r} for "
            f"{problem.name!r}; an optimal certificate was required"
        )
    return status


def _validated_partition_items(constraint, qmax):
    """Return native-integer partition inputs without coercing JSON booleans."""

    if not isinstance(constraint, dict):
        raise ValueError("partition constraint must be an object")
    if type(qmax) is not int or qmax < 0:
        raise ValueError("Qmax must be a non-negative integer")
    qbase = constraint.get("num_oq")
    if type(qbase) is not int or qbase < 0:
        raise ValueError("partition base-qubit count must be a non-negative integer")

    items = []
    for node, group in constraint.items():
        if node == "num_oq":
            continue
        if type(node) is not int:
            raise ValueError("partition constraint gate identities must be integers")
        if not isinstance(group, dict):
            raise ValueError("partition constraint gate entries must be objects")
        demand = group.get("num_eq")
        if type(demand) is not int or demand < 0:
            raise ValueError(
                "partition constraint demands must be non-negative integers"
            )
        if "qubits" in group:
            qubits = group["qubits"]
            if (
                not isinstance(qubits, list)
                or any(type(qubit) is not int for qubit in qubits)
            ):
                raise ValueError(
                    "partition constraint qubit identities must be integers"
                )
        items.append((node, demand))
    return qbase, items


def first_fit_decreasing_partition(constraint, qmax):
    """Return the deterministic FFD baseline and source-derived inventory."""

    qbase, items = _validated_partition_items(constraint, qmax)
    cap = qmax - qbase
    feasible = [(n, c) for n, c in items if c <= cap]
    infeasible = [n for n, c in items if c > cap]
    batches = []
    for n, c in sorted(feasible, key=lambda t: (-t[1], t[0])):
        for batch in batches:
            if batch['cost'] + c <= cap:
                batch['nodes'].append(n)
                batch['cost'] += c
                break
        else:
            batches.append({'nodes': [n], 'cost': c})
    for batch in batches:
        batch['nodes'].sort()
    inventory = {
        "capacity": cap,
        "item_count": len(items),
        "feasible_item_count": len(feasible),
        "infeasible_item_count": len(infeasible),
    }
    return batches, sorted(infeasible), inventory


def _canonical_batches(batches):
    return sorted(
        ({"nodes": list(batch["nodes"]), "cost": batch["cost"]}
         for batch in batches),
        key=lambda batch: tuple(batch["nodes"]),
    )


def validate_partition_certificate(
        constraint, qmax, batches, infeasible, metadata, *,
        ilp_node_limit=DEFAULT_ILP_NODE_LIMIT,
        refinement_time_limit_seconds=DEFAULT_REFINEMENT_TIME_LIMIT_SECONDS,
        gurobi_license_retry_delays_seconds=None):
    """Validate a stored achieved plan without repeating the timed MILP."""

    ffd_batches, expected_infeasible, inventory = (
        first_fit_decreasing_partition(constraint, qmax)
    )
    if not isinstance(batches, list) or not isinstance(infeasible, list):
        raise ValueError("partition certificate batches/infeasible must be lists")
    if not isinstance(metadata, dict):
        raise ValueError("partition certificate metadata must be an object")
    raw_nodes = [node for node in constraint if node != "num_oq"]
    demands = {}
    for node in raw_nodes:
        raw_demand = constraint[node].get("num_eq")
        if type(raw_demand) is not int or raw_demand < 0:
            raise ValueError("partition constraint demands must be non-negative integers")
        demands[node] = raw_demand
    feasible_nodes = set(demands) - set(expected_infeasible)
    if any(type(node) is not int for node in infeasible):
        raise ValueError("partition certificate infeasible identities must be integers")
    if infeasible != expected_infeasible:
        raise ValueError("partition certificate infeasible identities drifted")
    seen = set()
    normalized = []
    for position, batch in enumerate(batches):
        if not isinstance(batch, dict):
            raise ValueError(f"partition batch {position} is not an object")
        nodes = batch.get("nodes")
        cost = batch.get("cost")
        if (
            not isinstance(nodes, list)
            or not nodes
            or any(type(node) is not int for node in nodes)
            or nodes != sorted(set(nodes))
            or type(cost) is not int
            or cost < 0
        ):
            raise ValueError(f"partition batch {position} is malformed or empty")
        if seen.intersection(nodes):
            raise ValueError("partition certificate repeats an item")
        if set(nodes) - feasible_nodes:
            raise ValueError("partition certificate contains an infeasible item")
        expected_cost = sum(demands[node] for node in nodes)
        if cost != expected_cost:
            raise ValueError("partition certificate batch cost drifted")
        if inventory["capacity"] < cost:
            raise ValueError("partition certificate batch exceeds capacity")
        seen.update(nodes)
        normalized.append({"nodes": list(nodes), "cost": cost})
    if seen != feasible_nodes:
        raise ValueError("partition certificate does not cover every feasible item")

    base_expectations = {
        **inventory,
        "ffd_batch_count": len(ffd_batches),
        "final_batch_count": len(normalized),
    }
    for field, expected in base_expectations.items():
        actual = metadata.get(field)
        if type(actual) is not int or actual != expected:
            raise ValueError(f"partition metadata {field} drifted")
    actual_solver = metadata.get("solver")
    expected_solver = solver_provenance(
        gurobi_license_retry_delays_seconds=(
            gurobi_license_retry_delays_seconds
        )
    )
    if not isinstance(actual_solver, dict):
        raise ValueError("partition metadata solver provenance drifted")
    for field in ("threads", "seed"):
        expected = expected_solver[field]
        actual = actual_solver.get(field)
        if expected is not None and type(actual) is not int:
            raise ValueError("partition metadata solver provenance drifted")
    retry_delays = actual_solver.get("transient_license_retry_delays_seconds")
    if (
        not isinstance(retry_delays, list)
        or any(type(delay) is not int for delay in retry_delays)
    ):
        raise ValueError("partition metadata solver provenance drifted")
    expected_gap = expected_solver["relative_mip_gap"]
    if expected_gap is not None and type(
        actual_solver.get("relative_mip_gap")
    ) is not float:
        raise ValueError("partition metadata solver provenance drifted")
    if actual_solver != expected_solver:
        raise ValueError("partition metadata solver provenance drifted")
    refinement = metadata.get("refinement")
    if not isinstance(refinement, dict):
        raise ValueError("partition metadata lacks refinement evidence")
    if type(ilp_node_limit) is not int or ilp_node_limit < 0:
        raise ValueError("ILP node limit must be a non-negative integer")
    attempted = 1 < len(ffd_batches) and len(feasible_nodes) <= ilp_node_limit
    target = len(ffd_batches) - 1 if attempted else None
    fixed_refinement = {
        "attempted": attempted,
        "node_limit": ilp_node_limit,
        "target_batch_count": target,
        "time_limit_seconds": float(refinement_time_limit_seconds),
    }
    for field, expected in fixed_refinement.items():
        actual = refinement.get(field)
        if type(expected) is int and type(actual) is not int:
            raise ValueError(f"partition refinement {field} drifted")
        if actual != expected:
            raise ValueError(f"partition refinement {field} drifted")

    canonical_plan = _canonical_batches(normalized)
    canonical_ffd = _canonical_batches(ffd_batches)
    status = refinement.get("solver_status")
    resolved = refinement.get("resolved")
    fallback = refinement.get("fallback_used")
    improved = refinement.get("improved")
    if not attempted:
        if (
            status != "not-run"
            or resolved is not None
            or fallback is not False
            or improved is not False
            or canonical_plan != canonical_ffd
        ):
            raise ValueError("non-attempted refinement evidence is inconsistent")
    elif improved is True:
        if (
            status != "Optimal"
            or resolved is not True
            or fallback is not False
            or not 0 < len(normalized) <= target
        ):
            raise ValueError("successful refinement evidence is inconsistent")
    elif status == "Infeasible":
        if (
            improved is not False
            or resolved is not True
            or fallback is not False
            or canonical_plan != canonical_ffd
        ):
            raise ValueError("infeasible refinement evidence is inconsistent")
    else:
        if (
            status not in {"Not Solved", "Undefined"}
            or improved is not False
            or resolved is not False
            or fallback is not True
            or canonical_plan != canonical_ffd
        ):
            raise ValueError("unresolved refinement evidence is inconsistent")


def partition_nodes(
        constraint, qmax, ilp_node_limit=DEFAULT_ILP_NODE_LIMIT, *, return_metadata=False,
        refinement_time_limit_seconds=DEFAULT_REFINEMENT_TIME_LIMIT_SECONDS,
        gurobi_license_retry_delays_seconds=None):
    """
    constraint: dict like solve_qmon_for_circuit input, i.e.
        {'num_oq': int, node: {'num_eq': int, 'qubits': [...]}, ...}
    Returns (batches, infeasible): batches is a list of
        {'nodes': [...], 'cost': int}; infeasible lists nodes whose own
        extra-qubit demand already exceeds the budget.
    """
    if type(ilp_node_limit) is not int or ilp_node_limit < 0:
        raise ValueError("ILP node limit must be a non-negative integer")
    if refinement_time_limit_seconds <= 0:
        raise ValueError("refinement time limit must be positive")
    qbase, items = _validated_partition_items(constraint, qmax)
    cap = qmax - qbase
    feasible = [(n, c) for n, c in items if c <= cap]
    batches, infeasible, inventory = first_fit_decreasing_partition(
        constraint, qmax
    )
    k_ffd = len(batches)
    refinement = {
        "attempted": False,
        "node_limit": ilp_node_limit,
        "target_batch_count": None,
        "time_limit_seconds": float(refinement_time_limit_seconds),
        "solver_status": "not-run",
        "resolved": None,
        "fallback_used": False,
        "improved": False,
    }

    # exact refinement for small instances: can k_ffd - 1 bins work?
    if 1 < k_ffd and len(feasible) <= ilp_node_limit:
        k_try = k_ffd - 1
        refinement.update({"attempted": True, "target_batch_count": k_try})
        prob = pulp.LpProblem('batch_partition', pulp.LpMinimize)
        x = {(n, b): pulp.LpVariable(f'x_{n}_{b}', cat='Binary')
             for n, _ in feasible for b in range(k_try)}
        for n, _ in feasible:
            prob += pulp.lpSum(x[n, b] for b in range(k_try)) == 1
        for b in range(k_try):
            prob += pulp.lpSum(c * x[n, b] for n, c in feasible) <= cap
        prob += 0
        status = solve_problem(
            prob,
            time_limit=float(refinement_time_limit_seconds),
            gurobi_license_retry_delays_seconds=(
                gurobi_license_retry_delays_seconds
            ),
        )
        status_name = pulp.LpStatus.get(status, str(status))
        refinement["solver_status"] = status_name
        if status not in {pulp.LpStatusOptimal, pulp.LpStatusInfeasible}:
            # FFD has already produced a feasible deployment. The MILP only
            # asks whether one fewer bin is possible, so an unresolved solve
            # retains the certified FFD packing.
            refinement["resolved"] = False
            refinement["fallback_used"] = True
        else:
            refinement["resolved"] = True
        if status == pulp.LpStatusOptimal:
            packed = [{'nodes': [], 'cost': 0} for _ in range(k_try)]
            for n, c in feasible:
                for b in range(k_try):
                    if pulp.value(x[n, b]) > 0.5:
                        packed[b]['nodes'].append(n)
                        packed[b]['cost'] += c
                        break
            batches = [b for b in packed if b['nodes']]
            refinement["improved"] = True

    for b in batches:
        b['nodes'].sort()
    metadata = {
        "solver": solver_provenance(
            gurobi_license_retry_delays_seconds=(
                gurobi_license_retry_delays_seconds
            )
        ),
        **inventory,
        "ffd_batch_count": k_ffd,
        "final_batch_count": len(batches),
        "refinement": refinement,
    }
    if return_metadata:
        validate_partition_certificate(
            constraint,
            qmax,
            batches,
            sorted(infeasible),
            metadata,
            ilp_node_limit=ilp_node_limit,
            refinement_time_limit_seconds=refinement_time_limit_seconds,
            gurobi_license_retry_delays_seconds=(
                gurobi_license_retry_delays_seconds
            ),
        )
        return batches, sorted(infeasible), metadata
    return batches, sorted(infeasible)
