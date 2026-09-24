# RQ1 disturbance-law analysis

## Scope

`src/rq1_full_tightness.py` evaluates every acted gate-qubit location in the
310 circuits listed by `inputs/certified_circuits.txt`. No circuits or
locations are sampled.

Measurements, resets, barriers, delays, snapshots, and save directives do not
create analysis locations. A two-qubit instruction contributes one location
for each acted qubit.

## Quantity evaluated

After each eligible source instruction, the state is reshaped across the
selected qubit versus the rest of the system. The analysis computes
`det(rho_q)` from the two Schmidt amplitudes, inserts one forced QMon
measurement at that location, and calculates the fidelity with the
unmonitored final state after tracing out monitor ancillas.

For a pure state, unitary source circuit, and one monitor, the evaluated law
is

```text
1 - F = 3 det(rho_q).
```

Monitor outcomes are enumerated exactly; this experiment does not use a
finite number of shots. A result satisfies the law when the absolute residual
is at most `1e-12`.

## Resource and status accounting

The calculation uses a 24-qubit statevector limit. Every attempted location
is classified as successful, a law violation, a resource limit, a timeout,
unsupported, or an execution error. Non-evaluated locations remain in the
attempted denominator and are reported separately from the exact-evaluation
denominator.

The retained summary in `results/core/summary.json` reports 56,386 attempted
locations. All 56,386 satisfy the law; there are
no law violations. The remaining locations are explicitly reported as
resource limits or timeouts.

## Output

The compact numerical result is under `rq1` in
`results/core/summary.json`. Figure data are represented by
`results/figures/fig_entanglement_disturbance.pdf` and its PNG version;
`results/figures/figure_metadata.json` records the plotted count and source
checksums.

Run the focused tests from `reconstruction/` with:

```bash
PYTHONPATH=src python -m unittest -v tests/test_rq1_full_tightness.py
```
