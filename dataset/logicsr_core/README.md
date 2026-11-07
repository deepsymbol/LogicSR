# LogicSR-Core v1

A small, fixed evaluation subset of LogicSR: 370 truth tables, everything at
**at most 10 inputs and at most 10 outputs**.

| Folder | Content | Instances |
|---|---|---|
| `AN/` | `in5_out5` (and10/and20/and40) and `in10_out10` (and40/and80/and160), `noise_0` and `noise_0.05`, first 10 files per directory | 120 |
| `ANO/` | same cells as AN | 120 |
| `BBM/` | every BBM model inside the scale band (18 models), `noise_00` and `noise_05`, grouped by `inX_outY` | 36 |
| `IWLS/` | every IWLS instance inside the scale band, grouped by `inX_outY` | 94 |

Operators per suite: AN, IWLS -> `and`, `not`; ANO, BBM -> `and`, `or`, `not`.

## Manifest

`logicsr_core_v1.txt` is the frozen instance list. Its header records the
selection rule, the operator vocabulary and a SHA-256 of the list body. All
paths are relative to this folder (e.g. `AN/in5_out5/and10/noise_0/1.truth`).

`v1` is immutable: any change to the selection ships as `v2` with a new
manifest.

## Integrity

`SHA256SUMS` lists a SHA-256 for every truth table, so the data itself (not
only the instance list) can be verified. From this folder:

```bash
sha256sum -c SHA256SUMS
```
