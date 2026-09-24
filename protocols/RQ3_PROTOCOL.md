# RQ3 mutation experiment

## Population

RQ3 uses the 238 circuits in `inputs/rq3_circuits.txt` and `Qmax=24`.
`inputs/rq3_single_error_manifest.json` defines five seeds and three single-gate
mutations per circuit and seed, for 3,570 fixed mutation records. The mutated
instruction is drawn from circuits with at least three eligible one- or
two-qubit unitary instructions. Of the 3,570 mutations, 2,860 change the final
measured-output distribution and 710 are output-equivalent.

The mutation file stores each replacement explicitly. Validation replays
those replacements and checks the QASM contents, mutation identity, output
distance, and equivalence label; mutations are not redrawn.

## Methods

Each record contains six method results:

- QMon with deployed checkpoints grouped into batches;
- analytical per-checkpoint QMon, a diagnostic counterfactual that reads the
  exact mutant probability at each deployed checkpoint and executes no
  measure-reset-replay circuit;
- statistical assertions;
- projection assertions;
- dynamic-ancilla assertions; and
- the MQT debugger baseline.

QMon and the statistical assertion use 8,192 shots, familywise
`alpha=0.01`, and Bonferroni correction over their respective implemented
checkpoint sets. The exact two-sided binomial test flags only when its
p-value is strictly below the per-checkpoint threshold.

Each baseline checkpoint concerns one qubit: a Z marginal, a rank-one
projector, a one-qubit assertion ancilla, or MQT `assert-eq q[k]`. MQT uses a
similarity threshold of 0.99 and its combined flag includes both evaluated
state mismatches and unevaluable or refusal outcomes.

The same five-seed reporting rule is applied to the two MQT components and
their sum. The retained rates are `18.7 +/- 6.8%` state mismatches,
`40.4 +/- 6.6%` unevaluable or refusal outcomes, and `59.2 +/- 6.2%`
combined flags.

## Scope and denominators

`in_scope_full` means that the changed gate lies in the backward causal cone
of at least one selected QMon location. `in_scope_qmon` applies the same rule
to deployed locations only. The candidate-covered detection population uses
`in_scope_full`.

Reported detection rates are the unweighted mean and sample standard
deviation of the five per-seed rates. Parenthesized fractions are pooled
counts. Only records completed by every compared method enter paired rates;
incomplete outcomes would be reported separately, not counted as misses.

The final files are complete: all 2,860 non-output-equivalent mutations and
all 238 correct circuits have complete results for all six methods.

## Cost summaries

Costs are computed once per unique circuit, not weighted by mutations:

- batched QMon uses the number of deployment batches `B`;
- analytical per-checkpoint QMon uses the number of deployed checkpoint reads
  `K_qmon`; and
- separate-run assertion methods use their implemented checkpoint count `K`.

`K_qmon` includes eligible standard final pre-measurement reads as well as
deployed mid-circuit checkpoints. Each final read is counted once because it
is obtained from the circuit's existing terminal measurement; it does not
require an additional instrumented batch.

The retained mean/median/nearest-rank P75/maximum values are `30.0/1/14/853`
batch runs for batched QMon, `84.3/21/80/1,626` checkpoint reads for
analytical per-checkpoint QMon, and `92.1/21/89/1,626` circuit runs for the
separate-run assertion methods. With 238 circuits, nearest-rank P75 is the
179th sorted value.

## Files and commands

`results/rq3/rq3_mutant.pkl` contains the 3,570 mutations and
`results/rq3/rq3_falsealarm.pkl` contains the 238 correct circuits. Table
values are stored in `rq3_aggregate.json` and `rq3_aggregate.txt`; the miss
analysis and MQT outcome decomposition are stored beside them.

From `reconstruction/`, aggregate the complete files with:

```bash
QMON_MILP_SOLVER=gurobi python rq3_realexec_eval.py --aggregate \
  results/rq3/rq3_mutant.pkl results/rq3/rq3_falsealarm.pkl
```

Gurobi is required to reproduce the stored deployment batch counts. CBC is
sufficient for tests but can select a different valid packing for some
circuits.

The multiple-fault analysis uses nested one-, two-, and three-mutation
prefixes from the same manifest. Its summary is under `multiple_fault` in
`results/core/summary.json`.
