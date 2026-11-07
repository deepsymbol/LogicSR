# Synthetic MIMO Logic Networks

This directory contains the scripts and utilities for generating synthetic multi-input, multi-output (MIMO) logic networks. The code supports generating small networks, constructing larger networks by combining smaller ones, and visualizing or converting network files.

The generated benchmark data is stored in the `AN/` and `ANO/` directories under `synthetic_MIMO/`.

## Code Structure

```text
generation_scripts/
├── generate_net.py
├── net_combine.py
├── methods/
├── tools/
└── readme.md
```

The main components are organized as follows:

- **`generate_net.py`** generates small logic networks.
- **`net_combine.py`** constructs larger networks by merging previously generated small networks.
- **`methods/`** contains the network data structures and the algorithms for network generation and merging.
- **`tools/`** provides utilities for network visualization, file conversion, and truth-table processing.

The files with the `_xor.py` suffix implement a separate extended-gate variant that supports AND, OR, NOR, NAND, and XOR gates. This variant is independent of the standard `net_combine.py` workflow.

## Data Configurations

Both generators write output paths relative to the current working directory. To obtain the intended directory structure, run the scripts from `synthetic_MIMO/generation_scripts/`.

The two standard network configurations are:

- **AN:** Selected using `--use-aig`. Networks use AND gates with edge inversion.
- **ANO:** Selected by default when `--use-aig` is omitted. Networks support a combination of AND and OR gates.

Generated files are organized by input count, output count, and gate count. For example:

```text
AN/
└── in6_out6/
    └── and20/
```

Depending on the generation settings, each network directory may contain:

- `aag/`: circuit files;
- `tt/`: truth tables;
- `pic/`: HTML network visualizations.

When noise generation is enabled, the corresponding truth tables are stored in separate subdirectories under `tt/`, named according to their noise levels.
The `AN/` and `ANO/` directories also contain the packaged benchmark data used in the LogicSR dataset.

**Note.** Statistics should be calculated for the specific dataset of interest if you want to do precise characterization. Some statistics we provide in the Tables is an aggregated overview rather than a specific version description, do not use them directly for a specific set. 

## Usage

All commands below should be executed from the `synthetic_MIMO/generation_scripts/` directory.

### 1. Generate Small Networks

Run:

```bash
python generate_net.py --use-aig
```

The generation batch is controlled by the `N` and `combins` variables defined in `generate_net.py`. Modify these settings to change the number and configurations of the generated networks.

In the current script, the default configuration generates five networks with:

```python
combins = [(6, 6, 20)]
```

Each tuple specifies `(inputs, outputs, gates)`. The script generates AAG circuit files, truth tables, and HTML visualizations. The `--use-aig` option selects the AN configuration; without this option, the generator uses ANO.

### 2. Generate Larger Networks

To construct larger networks by combining smaller ones, run:

```bash
python net_combine.py --num_aigs 3 --tt_len 15 --batch_size 6 --multiple 5
```

The main command-line options are:

| Option | Description |
|---|---|
| `--use-aig` | Saves the generated networks under `AN/`. Without this option, the output is saved under `ANO/`. |
| `--num_aigs` | Number of merged networks to generate. |
| `--tt_len` | Exponent used to set the truth-table length limit. The implementation uses up to \(2^{\texttt{tt\_len}}\) rows, subject to the input-count constraint. |
| `--parallel` | Enables batched multiprocessing when saving generated results. |
| `--batch_size` | Number of generated networks saved in each batch. |
| `--multiple` | Oversampling factor used when selecting unique component networks. |
| `--rewrite` | Restarts the output numbering. This option is enabled by default in the current script. |

The target network configurations (`combins`) and component-network configurations (`small_aigs`) are defined in the `__main__` block of `net_combine.py`. Modify these tuples to change the network sizes and merging configurations.

The current settings are:

```python
combins = [(20, 20, 300)]
small_aigs = [(7, 7, 2, 120)]
```

The tuples specify:

- `combins`: `(inputs, outputs, gates)` for the target networks.
- `small_aigs`: `(inputs, outputs, component_count, component_gates)` for the component networks.

### 3. Generate Extended-Gate Networks

An independent generator is provided for the extended-gate variant. Run it with:

```bash
python methods/generate_net_xor.py
```

Its generation configurations are defined in the script. The implementation uses `methods/net_xor.py` and `methods/methods_plain_xor.py` and supports AND, OR, NOR, NAND, and XOR gates.

