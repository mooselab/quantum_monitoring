# Deriving the RQ2 table from the Qmax sweep

## Purpose

RQ2 uses the `Qmax=24` slice of the seven-budget sensitivity analysis. The
conversion is a projection of already computed rows; it does not select new
locations, repack monitor groups, or execute circuits.

## Required input

`src/rq2_from_qmax.py` requires one complete row for each of the 310 circuits and
each of the seven fixed budgets. Before selecting the `Qmax=24` rows, it
checks the circuit order, QASM identities, budget grid, row uniqueness,
planning settings, and all result checksums.

For each circuit, fields shared across budgets must agree, including source
width, source gate and depth counts, selected locations, final-read locations,
and monitor-group demands. Budget-dependent planning fields are retained from
the `Qmax=24` row.

## Output

The published circuit-level table is
`results/core/coverage.csv`. Its 310 rows contain the candidate and deployed
coverage values, batch and operation counts, structural predictors, and
source-circuit identifiers used by the RQ2 summaries and regression.

The complete raw seven-budget file is not included in the compact replication
package. The converter is retained to document and test the exact projection.
Use `PYTHONPATH=src python src/rq2_from_qmax.py --help` for its command-line
interface.
