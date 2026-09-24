"""Regression tests for explicit MILP backend selection and batching."""

from __future__ import annotations

import copy
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

SOURCE_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

import batch_partition as batching
import pulp.apis.gurobi_api as gurobi_api


def refinement_fixture():
    # FFD uses three capacity-10 bins, while two bins are feasible:
    # (6+2+2) and (5+3+2).
    costs = (6, 5, 3, 2, 2, 2)
    constraint = {"num_oq": 0}
    for node, cost in enumerate(costs):
        constraint[node] = {"num_eq": cost, "qubits": []}
    return constraint


class BackendSelectionTests(unittest.TestCase):
    def test_default_publication_backend_is_gurobi(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            batching.pulp, "GUROBI", wraps=batching.pulp.GUROBI
        ) as gurobi:
            batching.make_milp_solver(time_limit=30)
        gurobi.assert_called_once_with(
            msg=False,
            timeLimit=30,
            manageEnv=True,
            MIPGap=0.0,
            Threads=1,
            Seed=0,
        )

    def test_zero_mip_gap_uses_the_native_gurobi_parameter_path(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            solver = batching.make_milp_solver(time_limit=30)
            provenance = batching.solver_provenance()
        self.assertNotIn("gapRel", solver.optionsDict)
        self.assertIn("MIPGap", solver.solver_params)
        self.assertEqual(solver.solver_params["MIPGap"], 0.0)
        self.assertIs(type(solver.solver_params["MIPGap"]), float)
        self.assertEqual(provenance["relative_mip_gap"], 0.0)
        self.assertIs(type(provenance["relative_mip_gap"]), float)

    def test_native_zero_mip_gap_reaches_the_gurobi_model(self):
        model = mock.Mock()
        environment = mock.Mock()
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            gurobi_api.gp, "Env", return_value=environment
        ), mock.patch.object(
            gurobi_api.gp, "Model", return_value=model
        ):
            solver = batching.make_milp_solver(time_limit=30)
            solver.initGurobi()
            batching._close_managed_solver(solver)
        model.setParam.assert_any_call("MIPGap", 0.0)
        model.setParam.assert_any_call("Threads", 1)
        model.setParam.assert_any_call("Seed", 0)

    def test_managed_gurobi_environment_is_closed_after_solve(self):
        model = mock.Mock()
        environment = mock.Mock()
        solver = mock.Mock(manage_env=True, model=model, env=environment)
        solver.init_gurobi = True
        problem = mock.Mock()
        problem.solve.return_value = batching.pulp.LpStatusOptimal
        with mock.patch.object(batching, "make_milp_solver", return_value=solver):
            status = batching.solve_problem(problem)
        self.assertEqual(status, batching.pulp.LpStatusOptimal)
        model.dispose.assert_called_once_with()
        environment.dispose.assert_called_once_with()
        self.assertFalse(solver.init_gurobi)

    def test_failed_gurobi_initialization_is_made_destructor_safe(self):
        solver = mock.Mock(manage_env=True, model=None, env=None)
        solver.init_gurobi = True
        problem = mock.Mock()
        problem.solve.side_effect = RuntimeError("license handshake failed")
        with mock.patch.object(batching, "make_milp_solver", return_value=solver):
            with self.assertRaisesRegex(RuntimeError, "license handshake failed"):
                batching.solve_problem(problem)
        self.assertFalse(solver.init_gurobi)
        solver.close.assert_not_called()

    def test_transient_gurobi_session_failure_retries_with_fresh_solver(self):
        transient_type = type(
            "GurobiError",
            (RuntimeError,),
            {"__module__": "gurobipy._exception"},
        )
        events = []
        first_environment = mock.Mock()
        first_environment.dispose.side_effect = lambda: events.append("dispose")
        first = mock.Mock(
            manage_env=True, model=None, env=first_environment
        )
        first.init_gurobi = True
        second_model = mock.Mock()
        second_environment = mock.Mock()
        second = mock.Mock(
            manage_env=True, model=second_model, env=second_environment
        )
        second.init_gurobi = True
        problem = mock.Mock()
        problem.solve.side_effect = [
            transient_type("Too many sessions, 5 active sessions"),
            batching.pulp.LpStatusOptimal,
        ]
        with mock.patch.dict(
            os.environ, {batching.SOLVER_BACKEND_ENV: "gurobi"}, clear=True
        ), mock.patch.object(
            batching, "make_milp_solver", side_effect=[first, second]
        ), mock.patch.object(
            batching.time, "sleep", side_effect=lambda _delay: events.append("sleep")
        ) as sleep:
            status = batching.solve_problem(problem)
        self.assertEqual(status, batching.pulp.LpStatusOptimal)
        sleep.assert_called_once_with(
            batching.GUROBI_LICENSE_RETRY_DELAYS_SECONDS[0]
        )
        self.assertFalse(first.init_gurobi)
        self.assertEqual(events, ["dispose", "sleep"])
        second_model.dispose.assert_called_once_with()
        second_environment.dispose.assert_called_once_with()
        self.assertFalse(second.init_gurobi)

    def test_cleanup_failure_is_destructor_safe_and_attempts_all_objects(self):
        model = mock.Mock()
        environment = mock.Mock()
        model.dispose.side_effect = RuntimeError("model dispose failed")
        environment.dispose.side_effect = RuntimeError("env dispose failed")
        solver = mock.Mock(manage_env=True, model=model, env=environment)
        solver.init_gurobi = True
        problem = mock.Mock()
        problem.solve.return_value = batching.pulp.LpStatusOptimal
        with mock.patch.object(batching, "make_milp_solver", return_value=solver):
            with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
                batching.solve_problem(problem)
        model.dispose.assert_called_once_with()
        environment.dispose.assert_called_once_with()
        self.assertFalse(solver.init_gurobi)

    def test_solver_provenance_returns_an_independent_deep_copy(self):
        with mock.patch.dict(
            os.environ, {batching.SOLVER_BACKEND_ENV: "gurobi"}, clear=True
        ):
            first = batching.solver_provenance()
            first["transient_license_retry_delays_seconds"].append(999)
            second = batching.solver_provenance()
        self.assertNotIn(
            999, second["transient_license_retry_delays_seconds"]
        )

    def test_per_call_retry_policy_does_not_mutate_default_provenance(self):
        custom = (1, 2, 3)
        with mock.patch.dict(
            os.environ, {batching.SOLVER_BACKEND_ENV: "gurobi"}, clear=True
        ):
            bounded = batching.solver_provenance(
                gurobi_license_retry_delays_seconds=custom
            )
            default = batching.solver_provenance()
        self.assertEqual(
            bounded["transient_license_retry_delays_seconds"], list(custom)
        )
        self.assertEqual(
            default["transient_license_retry_delays_seconds"],
            list(batching.GUROBI_LICENSE_RETRY_DELAYS_SECONDS),
        )

    def test_solve_problem_uses_the_per_call_retry_policy(self):
        transient_type = type(
            "GurobiError",
            (RuntimeError,),
            {"__module__": "gurobipy._exception"},
        )
        first = mock.Mock(manage_env=False)
        second = mock.Mock(manage_env=False)
        problem = mock.Mock()
        problem.solve.side_effect = [
            transient_type("Too many sessions"),
            batching.pulp.LpStatusOptimal,
        ]
        with mock.patch.dict(
            os.environ, {batching.SOLVER_BACKEND_ENV: "gurobi"}, clear=True
        ), mock.patch.object(
            batching, "make_milp_solver", side_effect=[first, second]
        ), mock.patch.object(batching.time, "sleep") as sleep:
            status = batching.solve_problem(
                problem,
                gurobi_license_retry_delays_seconds=(7,),
            )
        self.assertEqual(status, batching.pulp.LpStatusOptimal)
        sleep.assert_called_once_with(7)

    def test_retry_policy_requires_native_nonnegative_integers(self):
        for invalid in ((1.0,), (-1,), (True,)):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, "non-negative integers"
            ):
                batching.solver_provenance(
                    gurobi_license_retry_delays_seconds=invalid
                )

    def test_nonlicense_gurobi_error_is_not_retried(self):
        error_type = type(
            "GurobiError",
            (RuntimeError,),
            {"__module__": "gurobipy._exception"},
        )
        solver = mock.Mock(manage_env=True, model=None, env=None)
        solver.init_gurobi = True
        problem = mock.Mock()
        problem.solve.side_effect = error_type("invalid model")
        with mock.patch.dict(
            os.environ, {batching.SOLVER_BACKEND_ENV: "gurobi"}, clear=True
        ), mock.patch.object(
            batching, "make_milp_solver", return_value=solver
        ), mock.patch.object(batching.time, "sleep") as sleep:
            with self.assertRaisesRegex(error_type, "invalid model"):
                batching.solve_problem(problem)
        sleep.assert_not_called()

    def test_invalid_backend_is_rejected(self):
        with mock.patch.dict(
            os.environ, {batching.SOLVER_BACKEND_ENV: "automatic"}, clear=True
        ):
            with self.assertRaisesRegex(ValueError, "unsupported"):
                batching.make_milp_solver()

    def test_backend_failure_never_falls_back(self):
        with mock.patch.dict(
            os.environ, {batching.SOLVER_BACKEND_ENV: "gurobi"}, clear=True
        ), mock.patch.object(
            batching, "make_milp_solver",
            side_effect=RuntimeError("license unavailable")
        ):
            with self.assertRaisesRegex(RuntimeError, "license unavailable"):
                batching.partition_nodes(refinement_fixture(), 10)


class CbcReplicationTests(unittest.TestCase):
    def test_partition_inputs_reject_boolean_integer_substitutes(self):
        valid = {
            "num_oq": 0,
            1: {"num_eq": 1, "qubits": [0]},
        }
        cases = {
            "qmax": (valid, True, batching.DEFAULT_ILP_NODE_LIMIT),
            "base qubit count": (
                {"num_oq": False, 1: {"num_eq": 1, "qubits": [0]}},
                2,
                batching.DEFAULT_ILP_NODE_LIMIT,
            ),
            "gate identity": (
                {"num_oq": 0, True: {"num_eq": 1, "qubits": [0]}},
                2,
                batching.DEFAULT_ILP_NODE_LIMIT,
            ),
            "ancilla demand": (
                {"num_oq": 0, 1: {"num_eq": True, "qubits": [0]}},
                2,
                batching.DEFAULT_ILP_NODE_LIMIT,
            ),
            "qubit identity": (
                {"num_oq": 0, 1: {"num_eq": 1, "qubits": [False]}},
                2,
                batching.DEFAULT_ILP_NODE_LIMIT,
            ),
            "ILP node limit": (valid, 2, True),
        }
        with mock.patch.dict(
            os.environ, {batching.SOLVER_BACKEND_ENV: "cbc"}, clear=True
        ):
            for label, (constraint, qmax, node_limit) in cases.items():
                with self.subTest(label=label), self.assertRaises(ValueError):
                    batching.partition_nodes(
                        constraint, qmax, ilp_node_limit=node_limit
                    )

    def test_certificate_rejects_boolean_plan_and_count_fields(self):
        constraint = {
            "num_oq": 0,
            1: {"num_eq": 1, "qubits": [0]},
        }
        with mock.patch.dict(
            os.environ, {batching.SOLVER_BACKEND_ENV: "cbc"}, clear=True
        ):
            batches, infeasible, metadata = batching.partition_nodes(
                constraint, 2, return_metadata=True
            )

            mutations = {
                "batch gate": lambda plan, _bad, _meta: plan[0]["nodes"].__setitem__(
                    0, True
                ),
                "batch cost": lambda plan, _bad, _meta: plan[0].__setitem__(
                    "cost", True
                ),
                "metadata item count": lambda _plan, _bad, meta: meta.__setitem__(
                    "item_count", True
                ),
                "provenance thread count": lambda _plan, _bad, meta: meta[
                    "solver"
                ].__setitem__("threads", True),
            }
            for label, mutate in mutations.items():
                plan = [dict(batch, nodes=list(batch["nodes"])) for batch in batches]
                bad = list(infeasible)
                meta = copy.deepcopy(metadata)
                mutate(plan, bad, meta)
                with self.subTest(label=label), self.assertRaises(ValueError):
                    batching.validate_partition_certificate(
                        constraint, 2, plan, bad, meta
                    )

            infeasible_constraint = {
                "num_oq": 2,
                1: {"num_eq": 1, "qubits": [0]},
            }
            plan, bad, meta = batching.partition_nodes(
                infeasible_constraint, 2, return_metadata=True
            )
            self.assertEqual(bad, [1])
            bad[0] = True
            with self.assertRaisesRegex(ValueError, "infeasible identities"):
                batching.validate_partition_certificate(
                    infeasible_constraint, 2, plan, bad, meta
                )

    def test_partition_certificate_rejects_empty_or_repartitioned_ffd_plan(self):
        constraint = {
            "num_oq": 0,
            1: {"num_eq": 3, "qubits": []},
            2: {"num_eq": 3, "qubits": []},
        }
        with mock.patch.dict(
            os.environ, {batching.SOLVER_BACKEND_ENV: "cbc"}, clear=True
        ):
            batches, infeasible, metadata = batching.partition_nodes(
                constraint, 10, return_metadata=True
            )
            batching.validate_partition_certificate(
                constraint, 10, batches, infeasible, metadata
            )

            with self.assertRaisesRegex(ValueError, "malformed or empty"):
                batching.validate_partition_certificate(
                    constraint,
                    10,
                    [*batches, {"nodes": [], "cost": 0}],
                    infeasible,
                    metadata,
                )

            split = [
                {"nodes": [1], "cost": 3},
                {"nodes": [2], "cost": 3},
            ]
            forged_metadata = dict(metadata)
            forged_metadata["final_batch_count"] = 2
            with self.assertRaisesRegex(ValueError, "non-attempted refinement"):
                batching.validate_partition_certificate(
                    constraint, 10, split, infeasible, forged_metadata
                )

    def test_explicit_cbc_refines_and_covers_every_item_once(self):
        with mock.patch.dict(
            os.environ, {batching.SOLVER_BACKEND_ENV: "cbc"}, clear=True
        ):
            batches, infeasible = batching.partition_nodes(
                refinement_fixture(), 10
            )
        self.assertFalse(infeasible)
        self.assertEqual(len(batches), 2)
        self.assertEqual(
            sorted(node for batch in batches for node in batch["nodes"]),
            list(range(6)),
        )
        self.assertTrue(all(batch["cost"] <= 10 for batch in batches))

    def test_partition_metadata_records_backend_and_refinement_certificate(self):
        with mock.patch.dict(
            os.environ, {batching.SOLVER_BACKEND_ENV: "cbc"}, clear=True
        ):
            batches, infeasible, metadata = batching.partition_nodes(
                refinement_fixture(), 10, return_metadata=True
            )
        self.assertFalse(infeasible)
        self.assertEqual(len(batches), 2)
        self.assertEqual(metadata["solver"]["backend"], "cbc")
        self.assertTrue(metadata["refinement"]["attempted"])
        self.assertEqual(metadata["refinement"]["solver_status"], "Optimal")
        self.assertTrue(metadata["refinement"]["resolved"])
        self.assertFalse(metadata["refinement"]["fallback_used"])
        self.assertTrue(metadata["refinement"]["improved"])
        self.assertEqual(metadata["final_batch_count"], 2)

    def test_unresolved_optional_refinement_retains_feasible_ffd(self):
        with mock.patch.dict(
            os.environ, {batching.SOLVER_BACKEND_ENV: "cbc"}, clear=True
        ), mock.patch.object(
            batching, "solve_problem", return_value=batching.pulp.LpStatusNotSolved
        ):
            batches, infeasible, metadata = batching.partition_nodes(
                refinement_fixture(), 10, return_metadata=True
            )
        self.assertFalse(infeasible)
        self.assertEqual(len(batches), metadata["ffd_batch_count"])
        self.assertEqual(metadata["final_batch_count"], 3)
        self.assertEqual(metadata["refinement"]["solver_status"], "Not Solved")
        self.assertFalse(metadata["refinement"]["resolved"])
        self.assertTrue(metadata["refinement"]["fallback_used"])
        self.assertFalse(metadata["refinement"]["improved"])

    def test_certificate_rejects_fabricated_status_and_noninteger_identity(self):
        with mock.patch.dict(
            os.environ, {batching.SOLVER_BACKEND_ENV: "cbc"}, clear=True
        ), mock.patch.object(
            batching, "solve_problem", return_value=batching.pulp.LpStatusNotSolved
        ):
            batches, infeasible, metadata = batching.partition_nodes(
                refinement_fixture(), 10, return_metadata=True
            )
            fabricated = mock.patch.dict(
                metadata["refinement"], {"solver_status": "fabricated-status"}
            )
            with fabricated, self.assertRaisesRegex(ValueError, "unresolved"):
                batching.validate_partition_certificate(
                    refinement_fixture(), 10, batches, infeasible, metadata
                )

            float_constraint = refinement_fixture()
            float_constraint[1.0] = float_constraint.pop(1)
            with self.assertRaisesRegex(ValueError, "identities must be integers"):
                batching.validate_partition_certificate(
                    float_constraint, 10, batches, infeasible, metadata
                )


if __name__ == "__main__":
    unittest.main()
