# RQ1 unfinished-location retries

## Scope

`inputs/rq1_retry_targets.json` lists exactly 11,136 historically unfinished
gate-qubit locations across 48 of the current 310 circuits: 6,409 resource
limits and 4,727 timeouts. Each target records its original point identity,
terminal status, record checksum, and source QASM checksum. The list has
1,413 tasks, each containing at most eight locations from one circuit.
Successful historical locations are excluded.

The source is a partial historical export containing 309 circuits, with file
SHA-256 `0f09185ee1dd875d5bea5ca627197354c8cd2aa19f4c7d0b660917ff874f720c`.
Its aggregate metadata are not the accepted 310-circuit baseline. The retry
list uses its individually checked circuit and point records only; the two
unfinished-status totals match the accepted RQ1 summary exactly. No final
merge should use this partial export as the complete baseline.

## Execution

`src/rq1_retry.py` uses the existing RQ1 local SVD calculation, forced-monitor
circuit construction, and exact branch-enumeration fidelity function. It
does not infer fidelity from the law being tested. New results are separate
from the original records. Existing output paths are rejected.

The historical run had a 24-qubit statevector cap and a six-hour circuit
timeout. The retry entry point defaults to a 27-qubit statevector cap and a
six-hour timeout per task of at most eight unfinished locations. These are
simulation resource limits, not a change to the deployment Qmax. A worker
writes each completed location immediately. If its task times out, remaining
requested locations retain an explicit timeout outcome.

A task can be run with an unused output filename:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  python src/rq1_retry.py run \
  --plan inputs/rq1_retry_targets.json --task-index 0 \
  --max-statevector-qubits 27 --timeout-seconds 21600 \
  --output results/rq1/retry_20260906/task_0000.json
```

Use a compute node with adequate memory for the 25--27-qubit statevectors
and their measurement branches. The command does not submit scheduler jobs.

## Pilot Result

On 2026-09-06, the single location
`canonical:078:ghz_indep_qiskit_13.qasm:g000012:q001` completed locally with
a 25-qubit cap and a 120-second timeout. It took 20.9 seconds, with fidelity
0.25, state infidelity 0.75, and law residual `3.33e-16`.

The result is `results/rq1/retry_20260906/ghz13_gate12_qubit1.json`.
There are 11,135 locations still awaiting retry. The original retry list is
kept unchanged. Task 169 contains the completed pilot point and one other
point; when scheduling the remaining work, run task 169 only with:

```text
--point-id canonical:078:ghz_indep_qiskit_13.qasm:g000012:q000
```

All other task indices retain their full listed scope. No cluster retry
array has been submitted, and the accepted RQ1 aggregate is unchanged.

Run the focused and related tests with:

```bash
PYTHONPATH=src QMON_MILP_SOLVER=cbc \
  python -m unittest discover -s tests -p 'test_rq1*.py'
```
