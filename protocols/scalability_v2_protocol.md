# Larger-circuit scalability analysis

## Retained data set

The replication package contains exactly 144 larger circuits with 15 to 50
qubits. Their complete records are in
`results/larger_circuits/larger_circuits_144.jsonl`; the compact plotting data
are in `larger_circuits_144_plot.csv`, and `manifest.json` lists all retained
cases and checksums. `structural_summary.json` records the reported selection
and structural-planning statistics in machine-readable form.

No additional generated circuits or unsuccessful source-generation attempts
are included. The 144 records are therefore the full retained data set for
this analysis, not a second circuit corpus competing with the 310-circuit
RQ1/RQ2 data set.

## Candidate-selection measurements

For each retained circuit, `scalability_v2.py` records:

- actual circuit width;
- MPS candidate-selection time;
- peak resident memory when available;
- maximum observed MPS bond dimension;
- selected gate-qubit locations and their causal-cone structure; and
- terminal selection and instrumentation status.

Candidate selection uses the same second Schmidt amplitude threshold as the
main QMon analysis. Figure 5 plots all 144 retained records; five completed
candidate selection but reached the later instrumentation time limit, so
they remain valid observations for selection time and bond dimension.

## Structural planning sweep

`larger_circuit_structural_sweep.py` evaluates the stored monitor structure at
`Qmax` values 60, 70, 80, 90, and 100. It does not rerun MPS selection and
does not execute monitored circuits.

Monitor groups are packed with deterministic FFD followed, for instances
with at most 120 groups, by one ILP feasibility check for one fewer batch.
The reported plan uses the improved packing when that check succeeds. The
procedure does not claim global optimality.

For each budget, the summary reports:

- fully deployable circuits;
- pooled selected-group and selected-location coverage;
- batch-count mean, median, 75th percentile, and maximum;
- peak ancilla use; and
- planned counted-operation multiplier.

The reported 75th percentile uses the nearest-rank definition. With 144
circuits, it is the 108th value after sorting in ascending order.

For one circuit, let `m` be the number of original unitary instructions, `B`
the number of planned batches, `G_replay` the total replayed source-gate
executions, and `M` the number of deployed monitor events. The multiplier is

```text
(B*m + G_replay + 2*M) / m.
```

The `2*M` term counts one monitor measurement and one reset per deployed
monitor. Source measurements, resets, barriers, and other directives are not
included in `m`.

## Validation and output

The inventory command checks that the retained file contains exactly 144
unique cases with the expected families and file checksum:

```bash
PYTHONPATH=src python src/larger_circuit_structural_sweep.py inventory \
  --source-artifact results/larger_circuits/larger_circuits_144.jsonl
```

Use `PYTHONPATH=src python src/larger_circuit_structural_sweep.py --help` for
per-record and summary commands. The retained reported values are in
`results/larger_circuits/structural_summary.json`. The plotted figure is available as
`results/figures/fig_scalability_mps.pdf` and PNG; its metadata are in
`results/figures/figure_metadata.json`.

Run focused tests from `reconstruction/` with:

```bash
QMON_MILP_SOLVER=cbc PYTHONPATH=src python -m unittest -v \
  tests/test_scalability_v2.py tests/test_larger_circuit_structural_sweep.py
```
