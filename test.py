import argparse
import os
import numpy as np
from evaluate.data_loader import load_truth_files
from evaluate.seed_config import DEFAULT_SEED
from evaluate.instance_timeout import run_instances
from experiment_runner import check_result_exists, run_experiment, load_method
from logicsr_core import SUITES
from logicsr_core.runner import run_suite


def main():
    parser = argparse.ArgumentParser(description='LogicBench Test Script')
    
    parser.add_argument(
        '-m', '--method', nargs='+', required=True, 
        help='Method names to run (refer to filenames in method folder). '
             'Multiple methods can be specified.'
    )
    parser.add_argument(
        '-d', '--data', nargs='+', default=None,
        help='Data path and optional subdirectory filters. '
             'First argument is the base path (e.g., ANO). '
             'Additional arguments filter files by subdirectory names. '
             'New shortcut: "-d ANO 5,10 0.01,0.05" selects all files under '
             'data/ANO with in5_out5 or in10_out10 and noise_0.01 or noise_0.05. '
             'Original filter mode is still supported. '
             'Required unless --suite is given.'
    )
    parser.add_argument(
        '--suite', choices=sorted(SUITES), default=None,
        help='Run a frozen, versioned instance list instead of -d filters. '
             '"core" reads logicsr_core/logicsr_core_v1.txt, which pins the '
             'exact instances and the per-dataset operator vocabulary so '
             'results stay comparable across papers. See logicsr_core/README.md.'
    )
    parser.add_argument(
        '-p', '--problem', nargs='+', default=[], 
        help='Logic operators to use (and, or, not). '
    )
    parser.add_argument(
        '-n', '--num-files', type=int, default=-1, 
        help='Number of files to randomly sample from the dataset. '
             'Use -1 to process all files (default). '
             'Use positive integer to limit the number of files.'
    )
    parser.add_argument(
        '-q', '--per-dir-limit', type=int, default=-1,
        help='For each innermost directory that contains .truth files, '
             'only take the first N files (sorted by numeric filename like 1.truth, 2.truth). '
             'Use -1 to take all files (default).'
    )
    parser.add_argument(
        '-s', '--split', type=float, default=0.75,
        help='Train/test split ratio for data splitting. '
             'Default is 0.75 (75%% training, 25%% testing). '
             'For small datasets (≤128 samples), full data is used for both train and test.'
    )
    parser.add_argument(
        '--seed', type=int, nargs='+', default=[DEFAULT_SEED],
        help='Random seed(s) for the stochastic part of each method. '
             'Give several to repeat every experiment once per seed, e.g. '
             '"--seed 0 1 2"; logicsr_core.aggregate then reports mean and '
             'spread across them. The train/test split is fixed regardless. '
             'Seed %(default)s keeps the original result file names.'
    )
    parser.add_argument(
        '-o', '--output-dir', default='output', 
        help='Output directory for saving results. '
             'Results are saved as JSON files in subdirectories mirroring the data structure.'
    )
    parser.add_argument(
        '--skip-existing', action='store_true',
        help='Skip instances that already have a result file. By default, every '
             'instance is run and an existing result file is overwritten.'
    )
    parser.add_argument(
        '--timeout', type=float, default=1200,
        help='Wall-clock limit per instance in seconds (default: %(default)s, as in '
             'the paper). Each instance runs in its own process; instances that '
             'exceed the limit are listed in timeouts.txt in the output directory. '
             'Use 0 to run all instances in this process without a limit.'
    )
    parser.add_argument(
        '--list-only', action='store_true', default=False,
        help='Only list the selected files and exit without running methods.'
    )
    parser.add_argument(
        '--no-use-ground-truth', dest='use_ground_truth', action='store_false',
        help='Evaluate accuracy against the provided (possibly noisy) truth table. '
             'By default, noisy files (e.g., noise_0.05) are evaluated against the '
             'corresponding noise_0 file.'
    )

    args = parser.parse_args()

    # A suite pins its own instances and operators; logicsr_core owns that path.
    if args.suite:
        run_suite(args)
        return

    if not args.data:
        parser.error('one of -d/--data or --suite is required')

    # Parse data path and subdirectory filters
    base_data_path = args.data[0]  # First parameter is base path
    subdir_filters = args.data[1:] if len(args.data) > 1 else []

    def _split_list_arg(raw: str):
        return [p.strip() for p in raw.split(',') if p.strip()]

    def _is_number_list(raw: str):
        parts = _split_list_arg(raw)
        if not parts:
            return False
        try:
            [float(p) for p in parts]
            return True
        except ValueError:
            return False

    # New optional mode: -d <dataset> <in_sizes> <noise_levels>
    # Example: -d ANO 5,10 0.01,0.05  → in5_out5 & in10_out10 with noise_0.01/0.05
    size_noise_mode = (
        len(subdir_filters) >= 2 and _is_number_list(subdir_filters[0])
        and _is_number_list(subdir_filters[1])
    )

    # Load data files
    data_files = load_truth_files(os.path.join('data', base_data_path))
    if not data_files:
        print(f" No .truth files found in data/{base_data_path}")
        return

    # If subdirectory filters are specified, filter files
    if size_noise_mode:
        in_sizes = [int(float(s)) for s in _split_list_arg(subdir_filters[0])]
        noise_levels_raw = _split_list_arg(subdir_filters[1])
        noise_filters = [
            lvl if lvl.startswith('noise_') else f"noise_{lvl}"
            for lvl in noise_levels_raw
        ]
        inout_filters = [f"in{n}_out{n}" for n in in_sizes]
        extra_filters = subdir_filters[2:] if len(subdir_filters) > 2 else []

        filtered_files = []
        for file_path in data_files:
            if not any(f in file_path for f in inout_filters):
                continue
            if not any(n in file_path for n in noise_filters):
                continue
            if extra_filters and not any(flt in file_path for flt in extra_filters):
                continue
            filtered_files.append(file_path)

        data_files = filtered_files
        print(
            f" Filtered (size/noise mode) to {len(data_files)} files: "
            f"in/out {in_sizes}, noise {noise_levels_raw}"
        )
    elif subdir_filters:
        filtered_files = []
        for file_path in data_files:
            # Check if file path contains any specified subdirectory filter
            if any(filter_name in file_path for filter_name in subdir_filters):
                filtered_files.append(file_path)
        data_files = filtered_files
        print(
            f" Filtered to {len(data_files)} files matching filters: {subdir_filters}"
        )

    if not data_files:
        print(f" No files found matching filters: {subdir_filters}")
        return

    # Limit number of files per innermost directory if requested
    if args.per_dir_limit is not None and args.per_dir_limit > 0:
        from pathlib import Path
        from collections import defaultdict

        def extract_numeric_filename(file_path: str) -> int:
            name = Path(file_path).name
            stem = name.split('.')[0]
            try:
                return int(stem)
            except Exception:
                return float('inf')

        files_by_dir = defaultdict(list)
        for fp in data_files:
            files_by_dir[str(Path(fp).parent)].append(fp)

        limited_files = []
        for dir_path, files in files_by_dir.items():
            files_sorted = sorted(files, key=extract_numeric_filename)
            limited_files.extend(files_sorted[:args.per_dir_limit])

        data_files = sorted(limited_files)

    # Randomly sample files if needed
    if args.num_files > 0 and args.num_files < len(data_files):
        data_files = np.random.choice(data_files,
                                      args.num_files,
                                      replace=False)

    print(f" Testing {len(data_files)} files from data/{base_data_path}")

    print(" Files to be processed:")
    print("-" * 60)
    for i, file_path in enumerate(data_files, 1):
        print(f"{i:3d}. {file_path}")
    print("-" * 60)

    if args.list_only:
        return

    # Run each method
    for method_name in args.method:
        print(f"\n Running method: {method_name}")
        operators = args.problem

        if args.timeout > 0:
            for seed in args.seed:
                if len(args.seed) > 1:
                    print(f"  seed {seed}")
                run_instances(method_name, data_files, operators, args, seed,
                              check_result_exists)
            continue

        method = load_method(method_name)
        if not method:
            continue

        if hasattr(method, 'set_operators'):
            method.set_operators(operators)

        from evaluate.operator_config import set_method_operators
        set_method_operators(method_name, operators)

        for seed in args.seed:
            if len(args.seed) > 1:
                print(f"  seed {seed}")

            run_experiment(method,
                           data_files,
                           split=args.split,
                           output_dir=args.output_dir,
                           method_name=method_name,
                           operators=operators,
                           skip_existing=args.skip_existing,
                           use_ground_truth=args.use_ground_truth,
                           seed=seed)
            # Results are now saved automatically within run_experiment

if __name__ == '__main__':
    main()