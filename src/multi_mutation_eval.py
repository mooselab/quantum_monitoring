"""
Multiple-error experiment (answers R4.6 "How does QMon work in the presence of
multiple errors?" and R4.7 "a multi-qubit gate error compiles into several
errors on different qubits").

Design (reuses the RQ3 v2 framework: same RQ3 circuit set, same
monitored-node semantics, same exact-marginal decision rules; skips the batch
MILP, which is irrelevant here):
  per circuit x seed (5 seeds) we inject
    m=1   one random unitary-gate mutation      (in-experiment baseline row)
    m=2   two simultaneous independent mutations (distinct gates)
    m=3   three simultaneous independent mutations
    clus  a CLUSTERED burst: one two-qubit gate plus up to 2 downstream unitary
          gates sharing its qubits within a 10-gate window, i.e. the compiled
          footprint of a single erroneous multi-qubit gate (R4.7).
Metrics per record (methods: same five as RQ3, exact shot->infinity limit):
  detected        any monitored node flags
  in_scope        >=1 injected fault lies in some monitored node's cone
  loc_any         earliest flagged node's cone contains >=1 injected fault
  loc_all         union of ALL flagged nodes' cones contains ALL faults
  equivalent      combined mutant leaves the output distribution unchanged
  masked          m>=2 only: every individual mutation alone is non-equivalent
                  but the COMBINATION is equivalent (errors cancel); these are
                  undetectable in principle by ANY output- or state-based
                  monitor, and are reported separately, not as misses.
"""
import sys
import random
import numpy as np

sys.setrecursionlimit(1000000)

from qiskit import QuantumCircuit
from mps_analysis import causal_cone
from mutation import mutate_circuit
from utilities import quantum_path, quantum_gates
from rq3_eval import (METHODS, monitored_nodes, marginals, per_node_flag,
                      final_probs, unitary_gate_indices)

WINDOW = 10          # clustered mode: downstream window (gate indices)


def _mutate_many(qc, gates, key="", seed=0):
    """Apply one mutation per gate index. The replacement drawn for gate g is
    seeded deterministically by (key, seed, g), so re-applying any SUBSET of
    `gates` (e.g. a single one, for the masking check) reproduces the exact
    same replacement for that gate as in the combined mutant."""
    import zlib
    mqc = qc
    for g in gates:
        random.seed(zlib.crc32(f"{key}|{seed}|{g}".encode()))
        mqc = mutate_circuit(mqc, g, quantum_gates)
    return mqc


def _cluster_gates(qc, cand, last_node, rng):
    """Pick a two-qubit gate g and up to 2 downstream unitary gates within
    WINDOW that touch g's qubits: the compiled footprint of one bad multi-qubit
    gate. Returns [] if the circuit has no eligible two-qubit gate."""
    twoq = [g for g in cand
            if qc.data[g].operation.num_qubits == 2 and g <= last_node]
    rng.shuffle(twoq)
    for g in twoq:
        qs = {qc.find_bit(b).index for b in qc.data[g].qubits}
        follow = [j for j in cand
                  if g < j <= min(g + WINDOW, last_node)
                  and qs & {qc.find_bit(b).index for b in qc.data[j].qubits}]
        if follow:
            return [g] + follow[:2]
    return []


def evaluate_circuit(key, seeds=(1, 2, 3, 4, 5)):
    qc = QuantumCircuit.from_qasm_file(quantum_path + key)
    nodes = monitored_nodes(qc)
    if not nodes:
        return []
    last_node = max(i for i, _ in nodes)
    if last_node <= qc.num_qubits - 1:
        return []
    measured = [qc.find_bit(ci.qubits[0]).index
                for ci in qc.data if ci.operation.name == "measure"]
    exp_rho = marginals(qc, nodes)
    base_probs = final_probs(qc, measured)
    cone = {nd: causal_cone(qc, nd[0], nd[1])[0] for nd in nodes}
    cand = [g for g in unitary_gate_indices(qc) if g <= last_node]
    if len(cand) < 3:
        return []

    records = []
    for seed in seeds:
        rng = random.Random(seed)
        random.seed(seed)                     # mutate_circuit uses global random
        variants = [("m1", rng.sample(cand, 1)),
                    ("m2", rng.sample(cand, 2)),
                    ("m3", rng.sample(cand, 3))]
        clus = _cluster_gates(qc, list(cand), last_node, rng)
        if len(clus) >= 2:
            variants.append(("clus", clus))
        for tag, gates in variants:
            try:
                mqc = _mutate_many(qc, gates, key=key, seed=seed)
                mp = final_probs(mqc, measured)
            except Exception:
                continue
            equivalent = bool(len(mp) == len(base_probs)
                              and np.abs(base_probs - mp).sum() < 1e-9)
            # masking: every fault individually changes the output, yet the
            # combination cancels
            masked = False
            if equivalent and len(gates) >= 2:
                try:
                    indiv = []
                    for g in gates:
                        sp = final_probs(_mutate_many(qc, [g], key=key, seed=seed),
                                         measured)
                        indiv.append(len(sp) == len(base_probs)
                                     and np.abs(base_probs - sp).sum() < 1e-9)
                    masked = not any(indiv)
                except Exception:
                    pass
            obs_rho = marginals(mqc, nodes)
            in_scope = any(g in cone[nd] for nd in nodes for g in gates)
            rec = {"circuit": key, "seed": seed, "tag": tag,
                   "gates": list(gates), "K": len(nodes),
                   "equivalent": equivalent, "masked": masked,
                   "in_scope": in_scope, "methods": {}}
            for m in METHODS:
                flagged = [nd for nd in nodes
                           if per_node_flag(m, exp_rho[nd], obs_rho[nd])[0]]
                detected = len(flagged) > 0
                first_cone = (cone[min(flagged, key=lambda nd: nd[0])]
                              if flagged else set())
                union_cone = set().union(*(cone[nd] for nd in flagged)) \
                    if flagged else set()
                rec["methods"][m] = {
                    "detected": detected,
                    "loc_any": any(g in first_cone for g in gates),
                    "loc_all": all(g in union_cone for g in gates),
                    "n_flagged": len(flagged),
                }
            records.append(rec)
    return records


def run(keys, out="multimut_records.pkl", quiet=False):
    import pickle, time
    recs, done, skipped = [], 0, 0
    t0 = time.perf_counter()
    for i, k in enumerate(keys):
        try:
            r = evaluate_circuit(k)
        except Exception as e:
            print(f"  ERR {k}: {str(e)[:60]}", flush=True)
            r = []
        if r:
            recs += r
            done += 1
        else:
            skipped += 1
        if out:
            with open(out, "wb") as f:
                pickle.dump(recs, f)
        if not quiet and (i + 1) % 20 == 0:
            print(f"  {done} ok / {skipped} skip / {len(keys)} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
    print(f"DONE: {done} circuits, {len(recs)} records "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)
    return recs


def aggregate(records):
    import statistics
    seeds = sorted({r["seed"] for r in records})

    def ms(vals, d=1):
        if not vals:
            return "n/a"
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return f"{np.mean(vals):.{d}f}+/-{sd:.{d}f}"

    print(f"\n===== multiple-error experiment: {len(records)} records, "
          f"{len({r['circuit'] for r in records})} circuits =====")
    for tag, label in [("m1", "1 error"), ("m2", "2 errors"),
                       ("m3", "3 errors"), ("clus", "clustered (R4.7)")]:
        sub = [r for r in records if r["tag"] == tag]
        if not sub:
            continue
        buggy = [r for r in sub if not r["equivalent"]]
        masked = [r for r in sub if r["masked"]]
        scoped = [r for r in buggy if r["in_scope"]]
        print(f"\n[{label}]  records {len(sub)}  non-equivalent {len(buggy)} "
              f" masked-cancellations {len(masked)}  in-scope {len(scoped)}")
        print(f"  {'method':12} {'detect%(all)':>13} {'detect%|in-scope':>17} "
              f"{'locAny%':>9} {'locAll%':>9}")
        for m in METHODS:
            det, dets, la, lu = [], [], [], []
            for s in seeds:
                b = [r for r in buggy if r["seed"] == s]
                bs = [r for r in scoped if r["seed"] == s]
                if b:
                    det.append(100 * np.mean([r["methods"][m]["detected"] for r in b]))
                if bs:
                    dets.append(100 * np.mean([r["methods"][m]["detected"] for r in bs]))
                dd = [r for r in b if r["methods"][m]["detected"]]
                if dd:
                    la.append(100 * np.mean([r["methods"][m]["loc_any"] for r in dd]))
                    lu.append(100 * np.mean([r["methods"][m]["loc_all"] for r in dd]))
            print(f"  {m:12} {ms(det):>13} {ms(dets):>17} "
                  f"{ms(la):>9} {ms(lu):>9}")


if __name__ == "__main__":
    import pickle
    if sys.argv[1:2] == ["--chunk"]:
        keys = [l.strip() for l in open(sys.argv[2]) if l.strip()]
        run(keys, out=sys.argv[3], quiet=True)
        sys.exit()
    if sys.argv[1:2] == ["--merge"]:
        merged = []
        for p in sys.argv[3:]:
            merged += pickle.load(open(p, "rb"))
        pickle.dump(merged, open(sys.argv[2], "wb"))
        aggregate(merged)
        sys.exit()
    if sys.argv[1:2] == ["--aggregate"]:
        aggregate(pickle.load(open("multimut_records.pkl", "rb")))
        sys.exit()
    keys = sys.argv[1:] or ["dj_indep_qiskit_5.qasm", "qpeinexact_indep_qiskit_5.qasm",
                            "wstate_indep_qiskit_5.qasm", "grover-v-chain_indep_qiskit_4.qasm",
                            "qft_indep_qiskit_6.qasm"]
    aggregate(run(keys, out=None))
