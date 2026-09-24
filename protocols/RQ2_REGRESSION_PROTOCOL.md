# RQ2 circuit-level regression

## Scope

`rq2_regression.py` analyzes conditional associations within the fixed set of
310 RQ2 circuits. It does not alter candidate selection, deployment planning,
or `Qmax`, and its coefficients are not interpreted causally.

The unit of analysis is one circuit. Exact zero and one coverage values are
retained.

## Outcomes

The two prespecified modeled outcomes are:

- deployed observable gate coverage; and
- deployed observable normalized-depth reach.

Deployed qubit coverage is reported descriptively but is not modeled because
its strong ceiling concentration leaves too little within-family variation
for the five-predictor family-aware model.

## Predictors

Both models contain these five predictors jointly:

- `E_mean`: mean `det(rho_q)` over all acted gate-qubit locations;
- `E_peak`: maximum `det(rho_q)` over those locations;
- `depth`: ASAP layer depth of the original, uninstrumented circuit, counting
  unitary instructions and excluding measurements, resets, barriers, delays,
  snapshots, and save directives;
- `two_qubit_density`: number of two-qubit unitary instructions divided by
  the total number of one- and two-qubit unitary instructions in the original
  circuit; and
- `num_qubits`: width of the original circuit.

Each predictor is centered by its full-sample mean and divided by its sample
standard deviation. The intercept is not standardized.

## Primary model

Each bounded outcome is fitted with fractional logit:

```text
E[y_i | x_i] = logistic(beta_0 + x_i beta).
```

The covariance is clustered by circuit family and includes the finite-cluster
correction `G/(G-1) * (n-1)/(n-p)`. Wald intervals and two-sided p-values use
the `t(G-1)` reference distribution.

As a distribution-free uncertainty check, complete circuit families are
resampled with replacement. The reported analysis uses 2,000 replicates and
seed `20260714`; at least 95% of requested fits must succeed. The predictor
scaling from the full sample is used in every resample.

The sensitivity model applies ordinary least squares to `asin(sqrt(y))` with
the same family-clustered covariance. It is a robustness comparison, not a
model-selection step.

## Multiple testing and diagnostics

The ten non-intercept tests from two outcomes and five predictors form one
family. Their raw p-values are adjusted together with the Benjamini-Hochberg
procedure.

The output also reports predictor correlations, variance inflation factors,
the design condition number, leverage, residuals, Cook distances, and
leave-one-family-out coefficient changes. These quantities diagnose model
stability; they do not determine which prespecified predictors are retained.

## Input and output

The analysis validates the complete coverage input before fitting and rejects
missing circuits, duplicate rows, inconsistent ratios, non-finite values, or
rank-deficient designs. `results/core/rq2_regression.json` contains the full
model output. `results/core/coverage.csv` contains the published input rows;
the larger JSON input used for the original validation is not included in the
compact package.

Run the tests with:

```bash
PYTHONPATH=src python -m unittest -v tests/test_rq2_regression.py
```
