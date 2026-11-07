# LogicSR: A Unified Benchmark for Logical Discovery from Data

LogicSR is a unified and scalable benchmark for logical symbolic regression. It combines real-world truth tables from logic circuits (IWLS) and biological networks (BBM) with a synthetic generator of multi-input, multi-output logic networks (MIMO), and it evaluates 18 methods from logic synthesis, symbolic regression, neural networks, and large language models.

This repository contains the evaluation framework, the method wrappers, the data generation scripts, and the LogicSR-Core evaluation set.

## Repository layout

| Path | Contents |
|---|---|
| `test.py` | Command-line entry point |
| `experiment_runner.py` | Runs a method on a list of instances and writes one JSON result per instance |
| `method/` | One wrapper per method |
| `evaluate/` | Data loading, metrics, gate counting, operator configuration, seeds |
| `external/` | Third-party code used by the wrappers; see the upstream projects for their licenses |
| `logicsr_core/` | LogicSR-Core: frozen instance list, runner, aggregation |
| `dataset/` | Generation and conversion scripts (see `dataset/README.md`) |

## Data


The full benchmark contains 3,244 networks and 6,132 truth tables.

Unpack and organize the data so that it has this layout:

```
data/
├── AN/ and ANO/                 # synthetic; AND/NOT and AND/OR/NOT
│   ├── in5_out5/and10/noise_0/1.truth
│   ├── in10_out10/and40/noise_0.05/1.truth
│   └── in20_out20/and160/1.truth
├── BBM/noise_00/easy/in5_out5/7.truth
└── IWLS/easy/in5_out8/1.truth
```

BBM and IWLS are further split by the number of inputs N into easy (N < 10), medium, and hard: medium is 10 ≤ N < 100 for BBM and 10 ≤ N ≤ 12 for IWLS; larger instances are hard.

Each `.truth` file stores one binary truth table. For `m` inputs and `n` outputs it has `m + n` rows: the first `m` rows are the input variables, and the last `n` rows are the output variables.

| Suite | Operators (`-p`) | Noise levels |
|---|---|---|
| AN | `and not` | 0, 0.01, 0.05 (`in5`, `in10`); 0 (larger scales) |
| ANO | `and or not` | 0, 0.01, 0.05 (`in5`, `in10`); 0 (larger scales) |
| BBM | `and or not` | `noise_00`, `noise_01`, `noise_05` |
| IWLS | `and not` | none |

## Installation

The experiments were run on Linux with a CUDA GPU for the neural methods.

```bash
conda create -n logicsr python
conda activate logicsr
pip install -r requirements.txt
```

To install the external dependencies in one step, run `bash install_external.sh`. The sections below describe the same steps for each method.

### ABC

```bash
cd external
git clone https://github.com/berkeley-abc/abc.git
cd abc
make ABC_USE_NO_READLINE=1
```

The wrapper expects the binary at `external/abc/abc`.

### PySR

The wrapper calls SymbolicRegression.jl v1.11.3 directly through PyJulia. The experiments used Julia 1.10.4.

```bash
conda install -c conda-forge julia
export JULIA_PROJECT=$PWD/julia_env
julia -e 'using Pkg; Pkg.add(name="SymbolicRegression", version="1.11.3"); Pkg.add("PyCall")'
```

### Boolformer

Boolformer uses the authors' pretrained weights without fine-tuning. Place them in `external/boolformer/`:

```bash
cd external/boolformer
gdown 1cULlE16yKBqUZMMGv7CC5fgHXXJ7OQQQ -O boolformer_noiseless.pt
gdown 1IFSc_sHfiTckMy-cwggvVMBH1YgXas8G -O boolformer_noisy.pt
```

### NeuraLUT

```bash
conda install -y packaging pyparsing
conda install -y docrep -c conda-forge
pip install --no-cache-dir git+https://github.com/Xilinx/brevitas.git@67be9b58c1c63d3923cac430ade2552d0db67ba5
pip install pyverilator==0.7.0
```

### DiffLogic

```bash
cd external/difflogic
python setup.py install
```

### SATNet

```bash
cd external
git clone https://github.com/locuslab/SATNet.git
cd SATNet
pip install -e .
```

This step needs PyTorch with CUDA and the `nvcc` compiler.


## Quick start

Run these commands from the repository root:

```bash
# One instance
python test.py -m sr_gplearn -d ANO/in5_out5/and10/noise_0/1.truth -p and or not

# One directory, several methods
python test.py -m sr_gplearn qm -d ANO/in5_out5/and10/noise_0 -p and or not

# List the selected files without running anything
python test.py -m sr_gplearn -d ANO 5,10 0.01,0.05 --list-only

# Noisy instances are scored against the clean (noise_0) truth table by default
python test.py -m sr_gplearn -d ANO/in5_out5/and10/noise_0.05 -p and or not

# Score against the provided noisy truth table instead
python test.py -m sr_gplearn -d ANO/in5_out5/and10/noise_0.05 -p and or not --no-use-ground-truth
```

### Methods

| Category | `-m` names |
|---|---|
| Logic synthesis | `abc`, `bdd`, `espresso`, `qm` |
| Symbolic regression | `sr_pysr`, `sr_operon`, `sr_deap`, `sr_gplearn`, `sr_tengp` |
| Neural networks | `nn_difflogic`, `nn_ai4eda`, `nn_neuralut`, `nn_boolformer`, `satnet` |
| LLMs | `gpt_4o`, `gpt_o3`, `gpt_4o_pot`, `gpt_o3_pot` |

`abc` applies only to AN and IWLS. `qm`, `bdd`, `espresso`, and `nn_boolformer` apply only to ANO and BBM.

### Main options

| Option | Meaning | Default |
|---|---|---|
| `-m` | Methods to run | required |
| `-d` | Data path under `data/`, with optional filters | required unless `--suite` is used |
| `--suite core` | Run LogicSR-Core | off |
| `-p` | Operators (`and`, `or`, `not`) | none |
| `-n` | Randomly sample this many files | all |
| `-q` | Take the first N files per directory | all |
| `-s` | Train fraction; tables of at most 128 rows use the full table for both | 0.75 |
| `--seed` | One or more seeds for the search; the split stays fixed | see `evaluate/seed_config.py` |
| `-o` | Output directory | `output` |
| `--timeout` | Wall-clock limit per instance in seconds; `0` disables it | 1200 |
| `--skip-existing` | Skip instances that already have results; otherwise they are run again and overwritten | off |
| `--no-use-ground-truth` | Score noisy instances against the provided noisy table, not the matching `noise_0` file | off |
| `--list-only` | List files and exit | off |


### Time limit

The paper uses a limit of 1200 s per instance, which is the default of `--timeout`. Each instance runs in its own process. An instance that exceeds the limit is stopped, writes no result file, and is listed in `timeouts.txt` in the output directory. Instances that fail are listed in `failed.txt`.

```bash
python test.py -m sr_operon -d ANO/in10_out10/and40/noise_0 -p and or not               # 1200 s per instance
python test.py -m sr_operon -d ANO/in10_out10/and40/noise_0 -p and or not --timeout 0   # no limit, one process
```

### Noise evaluation

Figure 4(c) scores against the clean truth table, which is the default. Figure 5(b) reports fidelity to the provided noisy truth table, which needs `--no-use-ground-truth`.

## LogicSR-Core

LogicSR-Core (v1) is a frozen set of 370 instances with at most 10 inputs and 10 outputs. It sets the operators per suite, so one command covers all four suites.

```bash
python test.py -m <method> --suite core                # one run
python test.py -m <method> --suite core --seed 0 1 2   # three seeds (recommended)
python -m logicsr_core.aggregate --output-dir output --by-cell
python -m logicsr_core.manifest --check                # verify the list against data/
```

See `logicsr_core/README.md` for the composition and the protocol.

## Output

Results mirror the data layout, for example `output/ANO/in5_out5/and10/noise_0/1/sr_gplearn.json`. Each file contains:

| Field | Meaning |
|---|---|
| `expression` | Learned expression(s) |
| `train_bit_acc`, `test_bit_acc` | Bit-wise accuracy |
| `train_sample_acc`, `test_sample_acc` | Sample-wise accuracy: all outputs of a row must match |
| `train_output_acc`, `test_output_acc` | Per-output correctness (POC): all rows of an output must match |
| `num_gates` | Complexity |
| `execution_time` | Wall-clock time in seconds |

Runs with a non-default seed are written to `<method>__seed<N>.json`.
