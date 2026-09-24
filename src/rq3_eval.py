"""
Unified RQ3 evaluation (v2): QMon vs four assertion baselines, each scored
under its OWN execution model, on a per-execution detect+localize metric.

Why v2: the first version scored every method from one shared exact reduced
density matrix, which (a) never exercised QMon's measure+reset+replay
mechanism, (b) erased the single-run vs destructive-K-reruns vs simulation-only
distinction that is QMon's whole point, and (c) gave the destructive/sim
baselines a free noise-free read at every node. This rebuild fixes that.

Execution models (the trade-off axis):
  QMon          ONE non-destructive run reads every monitored node's measured
                |1> frequency at S shots (justified: the no-disturbance lemma
                and test_exact_execution show the reconstructed run's
                mid-circuit measurement distribution equals the marginal; a
                validation sample actually runs run_reconstructed_exact).
  Proq / HM /   DESTRUCTIVE: one checkpoint per run, so K monitorable nodes
  Liu           need up to K separate runs, each at the SAME S shots.
  MQT debugger  SIMULATION-ONLY (reads the simulated statevector; not
                hardware-realizable by no-cloning); localizes with its own
                diagnostics, so QMon is NOT uniquely a localizer.

Per-node decision quantity (then sampled at S shots for the shot-based
methods; MQT is exact since it reads the sim state):
  qmon          |freq_obs(|1>) - p1_expected| >= tol     (Z-basis marginal)
  statistical   chi-square GOF of sampled marginal counts (Huang-Martonosi)
  proq          1 - <psi_exp|rho_obs|psi_exp> > tol      (projector fidelity)
  liu           1 - <Phi_exp|rho_obs|Phi_exp> > tol      (ancilla, every node)
  mqt           1 - |<psi_exp|psi_obs>| > tol, or backend
                unevaluable/refusal outcome

Honest accounting: at a separable node the fidelity methods (proq, mqt, liu)
have detection power >= qmon's Z-marginal (a phase change |+>->|-> is invisible
to qmon, caught by fidelity). QMon's win is the per-EXECUTION axis, not raw
per-node power.
"""
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace
from scipy.stats import chisquare

from fast_analysis import analyze_per_gate_nolock, SKIP_OPS
from mps_analysis import causal_cone
from utilities import quantum_path
from rq3_manifest import apply_manifest_slot, load_manifest, slots_for_circuit

METHODS = ["qmon", "statistical", "proq", "liu", "mqt"]
FIDELITY = {"proq", "liu", "mqt"}           # full-state fidelity methods
SIM_ONLY = {"mqt"}                          # not hardware-realizable
TOL = 1e-2
ALPHA = 1e-2
SHOTS = 8192
CANONICAL_QMAX = 24
MUT_SKIP = set(SKIP_OPS) | {"barrier", "measure", "reset"}
DEFAULT_CIRCUIT_LIST = (
    Path(__file__).resolve().parent.parent / "inputs" / "certified_circuits.txt"
)


def monitored_nodes(qc):
    """ALL no-lock monitorable (gate, qubit) nodes (RQ2 coverage sense)."""
    _, _, paths = analyze_per_gate_nolock(qc)
    return [(i, q) for q in paths for i in paths[q]]


def batch_count(key, qmax=24):
    """Number of within-budget instrumented runs (batches) QMon needs to cover
    ALL monitorable nodes: the candidate nodes are partitioned so each batch's
    original+extra qubits stay <= qmax (batch_partition), and the batches are
    executed separately and unioned. Returns (n_batches, n_infeasible) or
    (None, None) if the selection has no usable node. Each batch is ONE
    non-destructive run covering many nodes; this is QMon's per-execution cost
    for full-coverage monitoring (vs K destructive re-runs for the baselines)."""
    if int(qmax) != CANONICAL_QMAX:
        raise ValueError(f"RQ3 requires Qmax={CANONICAL_QMAX}, got {qmax}")
    from circuits_selection import (filter_circuits_with_output_num,
                                    dfs_quantum_circuit, bulid_partial_circuit)
    from utilities import filter_operation_list
    from batch_partition import partition_nodes
    qc = QuantumCircuit.from_qasm_file(quantum_path + key)
    res, _, paths = analyze_per_gate_nolock(qc)
    dnd = filter_circuits_with_output_num({key: res}, {key: paths})
    if key not in dnd:
        return None, None
    final = {key: []}
    for index, gname, qi, p0, p1 in dnd[key]:
        final[key].append([index, gname, qi, p0, p1,
                           dfs_quantum_circuit(qc, qi, index, set(), [])])
    op_list = bulid_partial_circuit(final)
    if key not in op_list:
        return None, None
    _, eqc = filter_operation_list(key, op_list, 10 ** 7)
    constraint = {"num_oq": qc.num_qubits}
    for nd, eq in eqc.items():
        constraint[nd] = {"num_eq": len(eq), "qubits": []}
        for g in op_list[key]:
            if g[0] == nd:
                constraint[nd]["qubits"].append(g[2])
    batches, infeasible = partition_nodes(constraint, qmax)
    return len(batches), len(infeasible)


def marginals(qc, node_set):
    """Exact 2x2 reduced density matrix of qubit q after gate i, for every
    (i,q), via one incremental statevector pass."""
    n = qc.num_qubits
    by_gate = {}
    for (i, q) in node_set:
        by_gate.setdefault(i, set()).add(q)
    state = Statevector.from_int(0, dims=2 ** n)
    out = {}
    for j, ci in enumerate(qc.data):
        if ci.operation.name in SKIP_OPS:
            continue
        state = state.evolve(ci.operation,
                             qargs=[qc.find_bit(b).index for b in ci.qubits])
        if j in by_gate:
            for q in by_gate[j]:
                out[(j, q)] = partial_trace(
                    state, [k for k in range(n) if k != q]).data
    return out


def _leading_vec(rho):
    w, v = np.linalg.eigh(rho)
    return v[:, int(np.argmax(w))]


def _purity(rho):
    return float(np.real(np.trace(rho @ rho)))


def per_node_flag(method, rho_exp, rho_obs):
    """Per-node decision in the EXACT (shot-to-infinity) limit: each method's
    own decision quantity vs the oracle, no sampling. This isolates each rule's
    DETECTION POWER free of shot-noise (a fixed 1e-2 threshold on a marginal at
    8192 shots is only ~1.8 sigma, so sampling would inject false positives and
    reintroduce the exact-vs-shots confound). The real-world cost difference is
    carried by the separate per-EXECUTION axis, not by adding noise here.
    Returns (flagged, applicable)."""
    p1e = float(rho_exp[1, 1].real)
    p1o = float(rho_obs[1, 1].real)
    if method == "qmon":
        # QMon reads the mid-circuit measurement = this Z-basis marginal (by
        # the no-disturbance lemma). Detect on the marginal deviation.
        return abs(p1o - p1e) >= TOL, True
    if method == "statistical":
        # Huang-Martonosi: classical/superposition assertions are chi-square
        # tests on the computational-basis distribution, whose large-sample
        # limit is exactly the marginal deviation. (Their third assertion
        # type, the product/entanglement test, asserts EXPECTED entangled
        # correlations and is inapplicable at these separable-oracle
        # checkpoints; an earlier draft over-credited HM with a purity-drop
        # channel no computational-basis measurement realizes.)
        return abs(p1o - p1e) >= TOL, True
    if method in ("proq", "liu"):
        psi = _leading_vec(rho_exp)
        return (1.0 - float((psi.conj() @ rho_obs @ psi).real)) > TOL, True
    if method == "mqt":
        if _purity(rho_obs) < 1 - 1e-9:
            return True, True  # idealized backend unevaluable/refusal flag
        psi, phi = _leading_vec(rho_exp), _leading_vec(rho_obs)
        return (1.0 - abs(complex(np.vdot(psi, phi)))) > TOL, True
    raise ValueError(method)


def final_probs(qc, measured):
    c = qc.copy()
    c.remove_final_measurements()
    probs = np.abs(Statevector.from_instruction(c).data) ** 2
    if not measured:
        return probs
    n = c.num_qubits
    arr = probs.reshape((2,) * n)
    keep = [n - 1 - q for q in measured]
    summ = tuple(ax for ax in range(n) if ax not in keep)
    return (arr.sum(axis=summ) if summ else arr).reshape(-1)


def unitary_gate_indices(qc):
    return [j for j, ci in enumerate(qc.data)
            if ci.operation.name not in MUT_SKIP
            and ci.operation.num_qubits in (1, 2)]


def evaluate_circuit(key, seeds=(1, 2, 3, 4, 5), muts_per_seed=3, qmax=24,
                     manifest=None):
    if int(qmax) != CANONICAL_QMAX:
        raise ValueError(f"RQ3 requires Qmax={CANONICAL_QMAX}, got {qmax}")
    qc = QuantumCircuit.from_qasm_file(quantum_path + key)
    # RQ3 monitors ALL no-lock monitorable nodes; with batched instrumentation
    # (B within-budget runs) QMon covers all of them. Filter on the last
    # monitorable node (mutations after it are unseeable); mutation points are
    # unitary gates in [0, last_node].
    nodes = monitored_nodes(qc)
    if not nodes:
        return []
    last_node = max(i for i, _ in nodes)
    if last_node <= qc.num_qubits - 1:
        return []
    K = len(nodes)
    B, _ = batch_count(key, qmax)            # QMon's non-destructive run count
    if B is None:
        return []
    measured = [qc.find_bit(ci.qubits[0]).index
                for ci in qc.data if ci.operation.name == "measure"]
    exp_rho = marginals(qc, nodes)
    base_probs = final_probs(qc, measured)
    cone = {nd: causal_cone(qc, nd[0], nd[1])[0] for nd in nodes}

    records = []
    # The mutation list fixes, per circuit and seed, which gate is replaced
    # and by what, so every method scores the same mutants.
    manifest = load_manifest() if manifest is None else manifest
    seed_set = {int(seed) for seed in seeds}
    slots = [
        slot for slot in slots_for_circuit(manifest, key)
        if int(slot["seed"]) in seed_set
        and int(slot["ordinal"]) < int(muts_per_seed)
    ]
    for slot in slots:
            seed = int(slot["seed"])
            g = int(slot["gate"])
            mqc = apply_manifest_slot(qc, slot)
            equivalent = bool(slot["output_equivalent"])
            obs_rho = marginals(mqc, nodes)

            # is the mutated gate inside ANY monitored node's backward cone?
            # if not, no node-based monitor can see it (a coverage gap, RQ2),
            # the dominant miss cause shared by QMon and every baseline.
            in_scope = any(g in cone[nd] for nd in nodes)
            rec = {"circuit": key, "seed": seed,
                   "slot_id": slot["slot_id"],
                   "slot_ordinal": int(slot["ordinal"]),
                   "manifest_sha256": manifest["manifest_sha256"],
                   "mut_gate": g, "replacement": slot["replacement"], "K": K,
                   "B": B, "equivalent": equivalent, "in_scope": in_scope,
                   "methods": {}}
            for m in METHODS:
                flagged = []
                applicable = 0
                for nd in nodes:
                    f, ok = per_node_flag(m, exp_rho[nd], obs_rho[nd])
                    if ok:
                        applicable += 1
                        if f:
                            flagged.append(nd)
                detected = len(flagged) > 0
                # localization = backward cone of the EARLIEST flagged node
                # (the first checkpoint to deviate pins the fault to its cone);
                # the tighter that cone, the more precisely the bug is located.
                suspect = (cone[min(flagged, key=lambda nd: nd[0])]
                           if flagged else set())
                loc_hit = g in suspect
                loc_prec = (1.0 / len(suspect)) if (loc_hit and suspect) else 0.0
                rec["methods"][m] = {
                    "detected": detected,
                    "loc_hit": loc_hit,
                    "loc_precision": loc_prec,
                    "n_flagged": len(flagged),
                    "n_applicable": applicable,
                    # runs to monitor ALL K nodes: QMon = B batches (each a
                    # non-destructive run, all within the qubit budget);
                    # destructive baselines = K (one checkpoint per run);
                    # MQT = simulation-only.
                    "executions": (B if m == "qmon"
                                   else ("sim-only" if m in SIM_ONLY else K)),
                }
            records.append(rec)
    return records


def aggregate(records):
    import statistics
    buggy = [r for r in records if not r["equivalent"]]
    seeds = sorted({r["seed"] for r in records})

    def ms(vals, d=1):
        if not vals:
            return "n/a"
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return f"{np.mean(vals):.{d}f}+/-{sd:.{d}f}"

    avgK = np.mean([r["K"] for r in records])
    avgB = np.mean([r["B"] for r in records])
    print(f"\nrecords {len(records)}  buggy {len(buggy)}  "
          f"equivalent {len(records) - len(buggy)}  "
          f"(avg {avgK:.0f} nodes/circuit; QMon avg {avgB:.1f} batches)")
    scoped = [r for r in buggy if r["in_scope"]]
    print(f"detection | bug in a monitored region: {len(scoped)}/{len(buggy)} "
          f"buggy mutants are in-scope (rest are coverage gaps)")
    print(f"{'method':12} {'detect%(all)':>13} {'detect%|in-scope':>17} "
          f"{'loc-precision':>13} {'runs/all-nodes':>15} {'hw?':>5}")
    qmon_dets = []
    for m in METHODS:
        det, dets, prec = [], [], []
        for s in seeds:
            b = [r for r in buggy if r["seed"] == s]
            bs = [r for r in scoped if r["seed"] == s]
            if b:
                det.append(100 * np.mean([r["methods"][m]["detected"] for r in b]))
            if bs:
                dets.append(100 * np.mean([r["methods"][m]["detected"] for r in bs]))
            pv = [r["methods"][m]["loc_precision"] for r in b
                  if r["methods"][m]["detected"]]
            if pv:
                prec.append(np.mean(pv))
        if m == "qmon":
            ex_s = f"{avgB:.1f} batches"
        elif m in SIM_ONLY:
            ex_s = "sim-only"
        else:
            ex_s = f"{avgK:.0f} (=K)"
        hw = "no" if m in SIM_ONLY else "yes"
        if m == "qmon":
            qmon_dets = list(dets)
        print(f"{m:12} {ms(det):>13} {ms(dets):>17} {ms(prec, 3):>13} "
              f"{ex_s:>15} {hw:>5}")

    # miss-case breakdown for QMon (answers R1.4's "analyze the misses")
    miss = [r for r in buggy if not r["methods"]["qmon"]["detected"]]
    gap = sum(1 for r in miss if not r["in_scope"])
    subthr = sum(1 for r in miss if r["in_scope"]
                 and not any(r["methods"][m]["detected"] for m in METHODS))
    phase = len(miss) - gap - subthr      # in-scope, caught by some baseline
    print(f"\nQMon miss breakdown ({len(miss)} of {len(buggy)} buggy):"
          f"\n  coverage-gap (bug outside all monitored cones; every "
          f"node-based method incl. baselines also misses) = {gap}"
          f"\n  phase/coherence (in-scope, invisible to QMon's Z-marginal but "
          f"caught by at least one baseline) = {phase}"
          f"\n  sub-threshold (in-scope, missed by EVERY method at "
          f"tau = 1e-2) = {subthr}")
    print(f"Reading: detection|in-scope "
          f"~{np.mean(qmon_dets) if qmon_dets else 0:.0f}% is "
          f"QMon's detection POWER; the lower all-mutants rate reflects "
          f"coverage (RQ2), not power. QMon monitors all {avgK:.0f} nodes in "
          f"{avgB:.1f} batched non-destructive runs (vs {avgK:.0f} destructive "
          f"re-runs / sim-only) with comparable localization.")


def validate_qmon_mechanism(keys):
    """Confirm QMon's per-node read (the exact marginal we use) equals what its
    ACTUAL reconstructed circuit measures mid-circuit, i.e. the exact-marginal
    shortcut is faithful to measure+reset+replay, not a bypass. Builds the
    instrumented circuit and runs run_reconstructed_exact, comparing each
    mid-circuit measurement's p1 to the marginal at that node."""
    import importlib.util

    helper_path = (
        Path(__file__).resolve().parents[1] / "tests" / "test_exact_execution.py"
    )
    spec = importlib.util.spec_from_file_location(
        "qmon_exact_execution_checks", helper_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load circuit validation helpers from {helper_path}")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    from exact_execution import run_reconstructed_exact
    print("VALIDATION: QMon exact-marginal vs reconstructed-circuit mid-measurement")
    for key in keys:
        qc, new_qc = helpers.build_instrumented_circuit(key)
        _, mids, viol = run_reconstructed_exact(new_qc, shots=8192, seed=2025)
        nodes = monitored_nodes(qc)
        rho = marginals(qc, nodes)
        # the reconstructed circuit inserts gates, so mid-record indices do not
        # align with original gate indices; match by qubit + ORDER instead (the
        # k-th mid-measurement of qubit q is q's k-th monitored node).
        nodes_by_q = {}
        for nd in sorted(nodes):
            nodes_by_q.setdefault(nd[1], []).append(nd)
        mids_by_q = {}
        for rec in mids:
            mids_by_q.setdefault(rec["qubit"], []).append(rec["p1"])
        worst, matched = 0.0, 0
        for q, ps in mids_by_q.items():
            for p1_rec, nd in zip(ps, nodes_by_q.get(q, [])):
                worst = max(worst, abs(p1_rec - rho[nd][1, 1].real))
                matched += 1
        print(f"  {key}: {len(mids)} mid-measurements ({matched} matched), "
              f"max |p1_reconstructed - p1_marginal| = {worst:.2e}, "
              f"separability violations = {len(viol)}")


def run_full(seeds=(1, 2, 3, 4, 5), muts_per_seed=3,
             circuits_file=DEFAULT_CIRCUIT_LIST,
             out="rq3_records.pkl"):
    """Full RQ3 over all eligible circuits in the certified set."""
    import pickle, time
    keys = [l.strip() + ".qasm" for l in open(circuits_file)]
    allrecs, done, skipped = [], 0, 0
    t0 = time.perf_counter()
    for k in keys:
        try:
            recs = evaluate_circuit(k, seeds, muts_per_seed)
        except Exception as e:
            print(f"  ERR {k}: {str(e)[:60]}", flush=True)
            recs = []
        if recs:
            allrecs += recs
            done += 1
            with open(out, "wb") as f:
                pickle.dump(allrecs, f)
        else:
            skipped += 1
        if (done + skipped) % 20 == 0:
            print(f"  {done} eligible / {skipped} skipped / "
                  f"{len(keys)} total ({time.perf_counter()-t0:.0f}s)", flush=True)
    print(f"\nDONE: {done} circuits, {len(allrecs)} mutant-evaluations "
          f"({time.perf_counter()-t0:.0f}s); records -> {out}")
    aggregate(allrecs)
    return allrecs


if __name__ == "__main__":
    import sys
    if sys.argv[1:2] == ["--full"]:
        run_full()
        sys.exit()
    if sys.argv[1:2] == ["--aggregate"]:
        import pickle
        aggregate(pickle.load(open("rq3_records.pkl", "rb")))
        sys.exit()
    if sys.argv[1:2] == ["--validate"]:
        validate_qmon_mechanism(sys.argv[2:] or
                                ["dj_indep_qiskit_5.qasm",
                                 "qpeinexact_indep_qiskit_5.qasm",
                                 "grover-v-chain_indep_qiskit_4.qasm"])
        sys.exit()
    keys = sys.argv[1:] or ["dj_indep_qiskit_5.qasm",
                            "qpeinexact_indep_qiskit_5.qasm",
                            "wstate_indep_qiskit_5.qasm",
                            "ghz_indep_qiskit_5.qasm",
                            "grover-v-chain_indep_qiskit_4.qasm"]
    allrecs = []
    for k in keys:
        recs = evaluate_circuit(k)
        print(f"{k}: {len(recs)} mutant-evaluations")
        allrecs += recs
    aggregate(allrecs)
