# LogicSR Core — frozen evaluation subset

The full LogicSR suite is large enough that few papers will run all of it, and
subsets chosen ad hoc are not comparable across papers. **Core** is a small,
versioned subset that is cheap to run and pinned by an explicit instance list,
so two papers reporting "LogicSR Core v1" are reporting on exactly the same
370 instances.

The instance list lives in `logicsr_core_v1.txt`. It is a plain list of paths,
not a set of filter arguments — filters are easy to mistype and their matching
semantics can change, a checked-in list cannot.

## The organising idea: one scale band

Core is everything the benchmark offers at **at most 10 inputs and at most 10
outputs**. The synthetic suites land in that band by construction (`in5_out5`,
`in10_out10`); the real-circuit suites are filtered down to it. That single
constraint is what makes the four suites comparable to each other, and it is
why the composition below needs no arbitrary "pick N models" rule for BBM and
IWLS.

## Layout

| File | Role |
|---|---|
| `__init__.py` | The declarative spec: scale band, suites, noise levels, operators |
| `logicsr_core_v1.txt` | The frozen instance list (370 paths + spec header + SHA-256) |
| `manifest.py` | Generates and verifies the list |
| `runner.py` | The `--suite` execution path `test.py` delegates to |
| `aggregate.py` | Reference table from existing per-instance results |

## Composition (v1)

**Synthetic — 240 instances.** A fixed number of lowest-index instances per
cell, matching what `test.py -q 10` selects.

| Suite | Input size | Gate counts | Noise | Per cell | Instances |
|---|---|---|---|---|---|
| AN | `in5_out5` | and10, and20, and40 | `noise_0`, `noise_0.05` | 10 | 60 |
| AN | `in10_out10` | and40, and80, and160 | `noise_0`, `noise_0.05` | 10 | 60 |
| ANO | `in5_out5` | and10, and20, and40 | `noise_0`, `noise_0.05` | 10 | 60 |
| ANO | `in10_out10` | and40, and80, and160 | `noise_0`, `noise_0.05` | 10 | 60 |

**Real circuits — 130 instances.** Every instance inside the scale band, so
there is no selection rule to argue about.

| Suite | Models in band | Noise | Instances |
|---|---|---|---|
| BBM | 18 (easy 13, medium 5) | `noise_00`, `noise_05` | 36 |
| IWLS | 94 (easy 64, medium 30) | none | 94 |

The scale band excludes every `hard` instance in both real suites — at this
scale the hard tier is empty, because those circuits are large by definition
(BBM `hard` runs to 371 inputs). Scaling and robustness claims belong to the
full suite, not to Core.

`noise_0` / `noise_0.05` (synthetic) and `noise_00` / `noise_05` (BBM) are the
two ends of each suite's noise range: the noiseless reference and the hardest
level. The intermediate level is left out to keep the runtime down; it remains
part of the full suite.

## Operator vocabulary

| Suite | Operators | Why |
|---|---|---|
| AN | `and`, `not` | AN instances are generated from AND/NOT only. Allowing OR lets a method search outside the space the data came from. |
| ANO | `and`, `or`, `not` | ANO instances use all three. |
| BBM | `and`, `or`, `not` | BBM circuits use all three. |
| IWLS | `and`, `not` | IWLS circuits are AND-inverter (AIG) style, built from AND/NOT. |

`--suite core` applies these automatically per suite, so one command covers all
four. Passing `-p/--problem` explicitly overrides them for every group.

## Protocol constants

| Constant | Value |
|---|---|
| Train/test split | 0.75, fixed split seed (datasets of ≤128 samples use the full data for both) |
| Reported accuracy | `bit_acc`, `output_acc`, `sample_acc` — train and test |
| Reported cost | `num_gates`, `execution_time` (wall clock, seconds) |
| Repeats | 3 seeds recommended; report the mean and the across-seed spread |

## Running it

```bash
python test.py -m <method> --suite core                # single run (seed 42)
python test.py -m <method> --suite core --seed 0 1 2   # three seeds
```

`-d`, `-n` and `-q` do not apply to a suite and are reported as ignored if
passed.

## Seeds and error bars

Because Core is cheap, the recommended protocol repeats each run over three
seeds and reports the mean and the across-seed spread — the run-to-run
variability that a single number hides.

`--seed` takes one or more seeds. Each reseeds the stochastic part of every
method (methods that expose no seed of their own draw from the global RNGs,
which are reseeded centrally) while the train/test split stays fixed, so what
varies is the search, not the data. Results are written to
`<method>__seed<N>.json`; the default seed keeps the original `<method>.json`
name, so runs made before seeds were configurable are read back unchanged.
`aggregate` then reports the mean over instances and the standard deviation
across seeds.

## Reference table

```bash
python -m logicsr_core.aggregate --output-dir output
python -m logicsr_core.aggregate --output-dir output --by-cell \
    --csv logicsr_core/logicsr_core_v1_reference.csv
```

Coverage is printed per method (`n/370`) and per metric, so a partially run
method is visible rather than silently averaged over whatever exists. When
several seeds are present the table adds the across-seed spread and the seed
count, so error bars come straight from the aggregation.

Prefer `--by-cell` when reporting. Core spans two noise levels and four suites,
and a single mean over all of them hides exactly the effects that separate
methods — for example `abc` scores 1.000 on AN `in5_out5` and 0.497 on AN
`in10_out10`, which a pooled average would erase.

## Integrity and versioning

`logicsr_core_v1.txt` carries the selection spec and a SHA-256 of its body in
the header. To confirm a checkout still matches the spec and the data tree:

```bash
python -m logicsr_core.manifest --check
```

The manifest is deterministic — no timestamps, no host paths — so regenerating
it on a correct checkout is a no-op.

**Versioning:** `v1` is immutable. Any change to the selection (different
instances, scale band, noise levels, or per-cell count) ships as `v2` with a
new manifest file; the old one stays so published numbers remain checkable.
