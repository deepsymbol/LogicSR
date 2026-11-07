# Dataset Overview

This directory provides truth-table datasets for symbolic regression and logic-network learning. It contains the curated **LogicSR-Core** evaluation subset, two real-world benchmark collections (BBM and IWLS), and synthetic MIMO networks generated with AN and ANO configurations.

## Updates

06/11/2026 The first release of LogicSR and LogicSR-Core.

## Directory Structure

```text
dataset/
├── logicsr_core/
│   ├── AN/
│   ├── ANO/
│   ├── BBM/
│   ├── IWLS/
│   ├── logicsr_core_v1.txt
│   ├── SHA256SUMS
│   └── README.md
├── real_world_BBM/
│   ├── scripts/
│   └── tseq_truth_tables/
├── real_world_IWLS/
│   ├── IWLS.zip
│   └── iwls_data_converter.py
├── synthetic_networks/
│   ├── AN/
│   ├── ANO/
│   └── generation_scripts/
└── README.md
```

## Dataset Collections

The repository contains four dataset collections:

| Dataset | Contents | Organization |
|---|---|---|
| **LogicSR-Core v1** | 370 truth tables, each with at most 10 inputs and 10 outputs | A fixed subset drawn from AN, ANO, BBM, and IWLS. The instance manifest and checksums are provided in [`logicsr_core/README.md`](logicsr_core/README.md). |
| **Synthetic MIMO** | Five ZIP archives for each of AN and ANO | Networks with 5, 10, 20, 40, and 80 inputs and outputs. Each archive contains circuit files, HTML visualizations, and truth tables. |
| **Real-world BBM** | 244 truth tables for each noise setting | Three ZIP archives corresponding to 0%, 1%, and 5% noise, with files grouped by input and output counts. |
| **Real-world IWLS** | 400 converted truth tables | A ZIP archive containing truth tables grouped by input and output counts. |

### Synthetic MIMO Networks

Synthetic MIMO networks are provided in two configurations:

- **AN:** Networks constructed using AND gates with edge inversion.
- **ANO:** Networks that additionally incorporate OR gates.

Each configuration contains the following five archives:

| Archive | Input/Output Size | Gate Counts |
|---|---:|---:|
| `in5_out5.zip` | 5 / 5 | 10, 20, 40 |
| `in10_out10.zip` | 10 / 10 | 40, 80, 160 |
| `in20_out20_tt15.zip` | 20 / 20 | 160, 320, 640 |
| `in40_out40_tt15.zip` | 40 / 40 | 320, 640, 1280 |
| `in80_out80_tt15.zip` | 80 / 80 | 640, 1280, 2560 |

Within each archive, files are organized as follows:

```text
in{N}_out{M}/
└── and{G}/
    ├── aag/    # Circuit files
    ├── pic/    # HTML network visualizations
    └── tt/     # Truth tables
```

The archives for the 5- and 10-input cases contain truth tables under three noise settings: `noise_0`, `noise_0.01`, and `noise_0.05`.

For the larger input sizes, the archives marked `tt15` contain truncated truth tables with at most \(2^{15}\) columns. The smaller input cases use complete truth tables.

The generation scripts and additional instructions are available in [`synthetic_networks/generation_scripts/readme.md`](synthetic_networks/generation_scripts/readme.md).

**ANO circuit format.** The `.aag` files in the ANO collection extend the standard AIGER-style gate record with a fourth field:

```text
lhs rhs0 rhs1 type
```

Here, `type = 0` represents an AND gate, while `type = 1` represents an OR gate. Edge inversion is encoded using odd literals, following the AIGER convention.

### Real-world BBM Networks

The BBM collection provides 244 truth tables under each of three noise settings. The corresponding archives are located in:

```text
real_world_BBM/tseq_truth_tables/
├── noise_00.zip
├── noise_01.zip
└── noise_05.zip
```

Each archive organizes `.truth` files into subdirectories named `in{N}_out{M}`, according to the number of inputs and outputs.

The scripts for model conversion and simulation are located in `real_world_BBM/scripts/`. They require the original BBM model files to be placed in a `models/` directory.

### Real-world IWLS Networks

The IWLS collection contains 400 converted truth tables in:

```text
real_world_IWLS/IWLS.zip
```

The files are organized into directories named `in{N}_out{M}`, where `N` and `M` denote the numbers of inputs and outputs, respectively.

To convert a source directory of IWLS truth tables, run the following command from the `dataset/` directory:

```bash
python real_world_IWLS/iwls_data_converter.py <input_dir> <output_dir>
```

The converter also supports the `--convert-existing` option, which reverses the bit order within each existing truth-table row in place under the input directory. Use this option only when the corresponding format conversion is required.

## Truth-Table Format

Each `.truth` file stores a binary matrix represented as text. Each row corresponds to one variable:

- The first \(N\) rows represent the inputs.
- The following \(M\) rows represent the outputs.
- Each character position corresponds to one truth-table assignment or sampled input configuration.

The directory name `in{N}_out{M}` indicates that the table contains \(N\) input rows and \(M\) output rows.

Depending on the dataset and input size, a truth table may be complete or truncated. A complete truth table with \(N\) inputs contains \(2^N\) columns, covering all possible input assignments. In contrast, the synthetic MIMO archives marked `tt15` contain at most \(2^{15}\) columns.

## Data Integrity

The LogicSR-Core v1 subset includes a fixed instance manifest and SHA-256 checksums to support reproducible evaluation.

To verify the integrity of the files, run the following command from the `logicsr_core/` directory:

```bash
sha256sum -c SHA256SUMS
```

## License

The code and the synthetic MIMO dataset are released under the MIT License. IWLS is released under the MIT License. The BBM dataset retains the original copyrights of the respective model authors and publishers.
