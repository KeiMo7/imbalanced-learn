# Sample-weight resampling report

Dataset: 400 rows, 6 features, classes {0: 100, 1: 300}, weight range [0.500, 4.972].

## Under-sampling keeps the original weights

| sampler | rows before | rows after | row-aligned | matches manual |
| --- | --- | --- | --- | --- |
| RandomUnderSampler | 400 | 200 | True | True |
| NearMiss | 400 | 200 | True | True |

## Over-sampling generated weights

| sampler | rows after | generated | originals preserved | generated range |
| --- | --- | --- | --- | --- |
| RandomOverSampler[inherit] | 600 | 200 | True | [2.073, 4.972] |
| SMOTE[inherit] | 600 | 200 | True | [2.073, 4.972] |
| SMOTE[nearest] | 600 | 200 | True | [2.079, 4.908] |
| SMOTE[cluster] | 600 | 200 | True | [3.561, 3.561] |

Weighted replication equivalence: total weighted = {0: 900.0, 1: 900.0}, replicated counts = {0: 900, 1: 900}, equivalent = True.

## Over-then-under chain

- rows: 600
- row-aligned: True
- matches explicit chain: True
- trainer sees resampled weights: True
- trainer fit matches direct fit: True

## Failure modes

| scenario | exception | message |
| --- | --- | --- |
| negative_weight | ValueError | RandomUnderSampler: 'sample_weight' contains 1 negative value(s) (min=-1, max=1, mean=0.995). Weights must be non-negati |
| length_mismatch | ValueError | RandomUnderSampler: 'sample_weight' has 399 rows while X and y have 400 rows. Weights must be aligned row-wise with the  |
| all_zero | ValueError | RandomUnderSampler: all 400 sample weights are zero. Refusing to silently fall back to uniform weights. |
| unsupported_strategy | InvalidParameterError | The 'weight_strategy' parameter of SMOTE must be a str among {'inherit', 'cluster', 'nearest'}, a callable or None. Got  |

## Unweighted path

- returns two outputs: True
- same random indices with and without weights: True

## Test suite

The pre-existing sampler test suites were run together with the new
weight test module; see ``pytest_samplers.txt`` for the raw output.

```
352 passed, 2 skipped (over_sampling, under_sampling, combine,
pipeline, base and sample_weight test modules)
```

One failure in ``imblearn/tests/test_common.py`` concerns
``BalancedRandomForestClassifier`` parameter constraints and is
pre-existing (it fails on the unmodified baseline as well).
