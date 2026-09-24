# QMon Replication Package

This repository accompanies *QMon: Monitoring the Execution of Quantum
Circuits with Mid-Circuit Measurement and Reset*. It contains the analyzed circuits, the research implementation, focused tests, and the result files used by the paper.

## Data Sets

- `quantum_circuits/` contains exactly the 310 QASM circuits analyzed in RQ1
  and RQ2. Their names are listed in
  `reconstruction/inputs/certified_circuits.txt`.
- `reconstruction/inputs/rq3_circuits.txt` lists the 238-circuit RQ3 subset.
- `reconstruction/inputs/rq3_single_error_manifest.json` fixes the 3,570 RQ3
  single-gate mutations: 238 circuits, five seeds, and three mutations per
  circuit and seed.
- `reconstruction/results/larger_circuits/` contains exactly the 144 retained
  larger-circuit records used for the 15 to 50 qubit scalability analysis.

Only the circuits, analysis code, tests, and result files needed to reproduce
the reported analyses are included.

## Repository Layout

```text
quantum_circuits/                 310 analyzed QASM circuits
reconstruction/
  src/                            analysis and baseline implementations
  tests/                          unit and integration tests
  protocols/                      experiment and validation protocols
  inputs/
    certified_circuits.txt        RQ1/RQ2 circuit list
    rq3_circuits.txt              RQ3 circuit list
    rq3_single_error_manifest.json
                                  fixed RQ3 mutations
  results/
    SHA256SUMS                    checksums for every retained result file
    core/                         RQ1, RQ2, Qmax, and multi-fault summaries
    rq1/                          additional unfinished-location retry results
    rq3/                          complete RQ3 records and summaries
    larger_circuits/              144 larger-circuit records
    figures/                      paper figures and plotting metadata
requirements.txt                  direct dependencies
requirements-lock.txt             complete environment lock
```

## Main Modules

| Module | Purpose |
| --- | --- |
| `src/fast_analysis.py` | Select monitorable gate-qubit locations from the separability test. |
| `src/circuits_selection.py` | Load circuits, collect causal cones, and construct monitored circuits. |
| `src/utilities.py` | Shared execution, grouping, replay, and ancilla-count helpers. |
| `src/exact_execution.py`, `src/verification.py` | Compare monitored execution with the original circuit. |
| `src/batch_partition.py` | Pack monitor groups into runs subject to the qubit budget. |
| `src/coverage_rq2.py` | Compute RQ2 candidate and deployed coverage. |
| `src/rq1_certificate_check.py`, `src/rq1_full_tightness.py` | Evaluate behavior preservation and the relation `1 - F = 3 det(rho_q)`. |
| `src/rq1_retry.py` | Retry only recorded resource-limited or timed-out RQ1 locations. |
| `src/rq2_regression.py` | Fit the two RQ2 fractional-logit models. |
| `src/mutation.py`, `src/rq3_manifest.py` | Define and validate the RQ3 mutations. |
| `src/rq3_realexec_eval.py` | Execute and aggregate QMon and the four baselines. |
| `src/rq3_miss_analysis.py` | Partition QMon misses by monitoring coverage and baseline outcome. |
| `src/mqt_diagnostics.py` | Split MQT flags into state mismatches and unevaluable outcomes. |
| `src/multi_mutation_realexec_eval.py` | Evaluate one, two, and three simultaneous mutations. |
| `src/qmax_sensitivity.py`, `src/rq2_from_qmax.py` | Evaluate and summarize the RQ2 `Qmax` sweep. |
| `src/scalability_v2.py`, `src/larger_circuit_structural_sweep.py` | Analyze MPS selection and structural overhead on the 144 larger circuits. |

The Markdown files under `reconstruction/protocols/` give the exact protocol
for each reported experiment.

## Environment

The final environment used Python 3.12.13. Create a clean environment with:

```bash
conda create -n qmon2 python=3.12 -y
conda activate qmon2
pip install -r requirements.txt
```

`requirements-lock.txt` records the complete Linux environment used for the
final runs. PuLP constructs the packing problem. Set
`QMON_MILP_SOLVER=gurobi` to reproduce the stored deployment plans; CBC is
sufficient for the test suite and small examples.

All commands below are run from `reconstruction/`.

## Tests

```bash
QMON_MILP_SOLVER=cbc PYTHONPATH=src \
  python -m unittest discover -s tests -p "test_*.py"
```

The tests cover circuit construction, replay, batch planning, RQ1 and RQ2
calculations, mutation identity, all baseline interfaces, RQ3 aggregation,
miss analysis, and the 144-circuit structural sweep.

## Planning Procedure

The reported plans use deterministic first-fit decreasing (FFD). For instances
with at most 120 monitor groups, one feasibility ILP checks whether the same
groups fit in one fewer batch. If it succeeds, that refined plan is used. This
is a one-step improvement, not a claim of globally optimal bin packing.

The main RQ1, RQ2, and RQ3 experiments use `Qmax=24`. The separate
larger-circuit structural sweep uses `Qmax` in `{60, 70, 80, 90, 100}` and the
same FFD plus one-step ILP procedure.

## Results

### RQ1 and RQ2

`results/core/summary.json` contains the compact RQ1, RQ2, RQ2 regression,
`Qmax`, and multiple-fault summaries. `results/core/coverage.csv` contains one
RQ2 row for each of the 310 circuits, and
`results/core/rq2_regression.json` contains the fitted regression results.

The principal RQ1 and RQ2 programs are:

```bash
PYTHONPATH=src python src/rq1_certificate_check.py
PYTHONPATH=src python src/qmax_sensitivity.py --help
PYTHONPATH=src python src/rq2_from_qmax.py --help
PYTHONPATH=src python src/rq2_regression.py --help
```

The additional RQ1 retry list in `inputs/rq1_retry_targets.json` contains
11,136 unfinished locations from 48 circuits. One location has completed in
the 2026-09-06 local pilot, leaving 11,135 awaiting retry; the accepted
summary has not been replaced. See `protocols/RQ1_RETRY_PROTOCOL.md` for
resource limits, provenance, and the exact remaining task scope.

### RQ3

`results/rq3/rq3_mutant.pkl` contains all 3,570 mutation records and
`results/rq3/rq3_falsealarm.pkl` contains all 238 unmutated circuits. Each file
contains QMon, analytical per-checkpoint QMon, statistical assertion,
projection assertion, dynamic ancilla, and MQT results.

Regenerate the RQ3 table values with:

```bash
QMON_MILP_SOLVER=gurobi PYTHONPATH=src \
  python src/rq3_realexec_eval.py --aggregate \
  results/rq3/rq3_mutant.pkl results/rq3/rq3_falsealarm.pkl
```

This command validates every stored deployment plan before aggregation and can
take substantial time. CBC may choose a different number of batches for a
large circuit and therefore must not be used to validate these stored files.

Regenerate the miss analysis and MQT outcome counts with:

```bash
PYTHONPATH=src python src/rq3_miss_analysis.py \
  results/rq3/rq3_mutant.pkl results/rq3/rq3_miss_analysis.json
PYTHONPATH=src python src/mqt_diagnostics.py \
  results/rq3/rq3_mutant.pkl results/rq3/mqt_diagnostics.json
```

`results/rq3/rq3_deployed_miss_diagnosis.json` contains the exact
checkpoint-distribution diagnosis for all 432 deployment-covered QMon misses:
262 have no raw deployed Z signal, 138 have a raw signal that is erased by the
emitted measure-reset-replay channel, and 32 retain a signal with low
finite-shot detection power.

The retained MQT diagnostics contain 2,860 complete non-output-equivalent
records: 534 evaluated state mismatches, 1,154 unevaluable or refusal outcomes,
and 1,688 combined flags. Their five-seed rates are respectively
`18.7 +/- 6.8%`, `40.4 +/- 6.6%`, and `59.2 +/- 6.2%`.

### Larger Circuits

`results/larger_circuits/larger_circuits_144.jsonl` contains the 144 retained
MPS selection records. `larger_circuits_144_plot.csv` is the compact plotting
table, `manifest.json` lists every retained case and checksum, and
`structural_summary.json` records the reported selection and `Qmax` planning
statistics.

Check the exact 144-record input with:

```bash
PYTHONPATH=src python src/larger_circuit_structural_sweep.py inventory \
  --source-artifact results/larger_circuits/larger_circuits_144.jsonl
```

The structural sweep reads these precomputed node-selection records. It does
not rerun MPS selection and does not execute monitored circuits. Use
`PYTHONPATH=src python src/larger_circuit_structural_sweep.py --help` for the
per-record and aggregation commands.

### Figures

`results/figures/` contains PDF and PNG versions of the RQ1
entanglement-disturbance plot and the larger-circuit MPS plot.
`figure_metadata.json` records the plotted sample counts and source-data
checksums.

## Data Source

The circuits are derived from [MQT Bench](https://www.cda.cit.tum.de/mqtbench/)
and decomposed to the one- and two-qubit basis used by the experiments.
