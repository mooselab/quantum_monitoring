# RQ3 miss analysis

## Scope

`src/rq3_miss_analysis.py` describes QMon misses in the complete RQ3 mutant
result. It does not change the mutation list, method implementations, or
finite-shot outcomes.

The population contains the 2,860 non-output-equivalent mutations completed
by QMon and all four reported baselines. The analytical per-checkpoint QMon
row is excluded from this common denominator because it is a diagnostic
counterfactual rather than a separately executed method.

## Rates

Seeds 1 through 5 remain separate. For every overall result and stratum, the
analysis reports attempts, detections, misses, and the QMon miss rate for each
seed. It then reports:

- the unweighted mean and sample standard deviation of the five seed rates;
- the pooled rate from the corresponding counts; and
- a 95% Wilson interval for the pooled rate.

These summaries answer different questions and must not be interchanged.

## Categories

Misses are summarized by circuit family, original and replacement gate
arity, operation type, and three structural scope classes:

- `outside-full-scope`: the changed gate cannot affect any selected QMon
  location;
- `deployment-gap`: an affected selected location exists but is not deployed
  under `Qmax=24`; and
- `in-scope-qmon`: the changed gate can affect at least one deployed QMon
  location.

The final data contain 1,078 QMon misses: 635 outside full scope, 11 in a
deployment gap, and 432 with at least one affected deployed location but no
QMon flag. The 432-case category is an observed outcome, not a claim about
its physical cause.

Strata with fewer than 20 observations or fewer than five represented seeds
are marked sparse. No between-stratum hypothesis tests are performed.

## Output

The retained result is `results/rq3/rq3_miss_analysis.json`, derived from
`results/rq3/rq3_mutant.pkl`. Recompute it with:

```bash
PYTHONPATH=src python src/rq3_miss_analysis.py \
  results/rq3/rq3_mutant.pkl \
  results/rq3/rq3_miss_analysis.json
```

Run the focused tests with:

```bash
PYTHONPATH=src python -m unittest -v tests/test_rq3_miss_analysis.py
```
