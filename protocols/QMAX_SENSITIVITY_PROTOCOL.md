# Qmax sensitivity analysis

## Scope

This analysis applies the RQ2 planning procedure to all 310 circuits in
`inputs/certified_circuits.txt` at `Qmax` values 16, 20, 24, 28, 32, 36,
and 40.
The resulting 2,170 circuit-budget combinations are summarized under `qmax`
in `results/core/summary.json`. The `Qmax=24` rows are also provided as
`results/core/coverage.csv`.

## Planning procedure

For each circuit and budget:

1. Candidate gate-qubit locations are selected with the second Schmidt
   amplitude criterion used by the main QMon analysis.
2. A selected location at the final source gate of a terminally measured
   qubit is covered by the circuit's final measurement and needs no inserted
   monitor.
3. Selected operands of the same source gate form one monitor group. Shared
   causal-cone operations are replayed once within that group.
4. A group is individually feasible only when the source width plus its
   required ancillas does not exceed `Qmax`.
5. Feasible groups are packed with deterministic first-fit decreasing (FFD).
   For instances with at most 120 groups, one ILP feasibility check tests
   whether all groups fit in one fewer batch. A successful check replaces the
   FFD plan; otherwise the FFD plan is retained. This is a one-step
   refinement, not a claim of globally optimal packing.
6. Every planned monitored circuit is constructed and checked against the
   budget, monitor-group membership, replay operations, ancilla allocation,
   and selected-location accounting.

Because the optional one-step refinement is time limited, the resulting
batch count is not assumed to be monotone in `Qmax`.

## Reported quantities

The summary reports, for each budget:

- selected locations and monitor groups;
- feasible and infeasible locations and groups;
- deployed gate, gate-qubit, qubit, and depth coverage;
- fully deployable circuit count;
- batch and deployment-run counts;
- peak width and ancilla use;
- replay, measurement, reset, and total planned operation counts; and
- source and monitored-circuit depth.

Depth counts executable operations, including measurements and resets, while
excluding barriers, delays, snapshots, and save directives.

## Validation

`src/qmax_sensitivity.py` checks the fixed circuit list, QASM contents, complete
budget grid, unique circuit-budget identities, and all stored planning
quantities. Its full validation mode recomputes candidate selection, causal
cones, group feasibility, packing validity, and monitored-circuit structure.
It does not repeat the optional time-limited ILP check.

Use `PYTHONPATH=src python src/qmax_sensitivity.py --help` from
`reconstruction/` for the available commands.
