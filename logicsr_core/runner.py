"""Execution path for a frozen suite, kept out of test.py.

test.py delegates here as soon as --suite is given: this module owns reading
the manifest, splitting it per dataset so each group runs with the operator
vocabulary the suite pins for it, and driving run_experiment.
"""

import os

from logicsr_core import SUITES


def load_suite(suite_name):
    """Return the instance paths of a suite, failing loudly on a bad manifest."""
    manifest = SUITES[suite_name]['manifest']
    if not os.path.exists(manifest):
        raise SystemExit(
            " Suite manifest not found: %s\n"
            " Generate it with: python -m logicsr_core.manifest" % manifest)

    with open(manifest, encoding='utf-8') as handle:
        files = [line.strip() for line in handle
                 if line.strip() and not line.startswith('#')]
    if not files:
        raise SystemExit(" Suite manifest is empty: %s" % manifest)

    missing = [path for path in files if not os.path.exists(path)]
    if missing:
        raise SystemExit(
            " Suite '%s' lists %d instance(s) that are not present, e.g. %s\n"
            " Run from the repository root, or regenerate the manifest."
            % (suite_name, len(missing), missing[0]))
    return files


def group_by_dataset(data_files, operators_by_dataset, fallback):
    """Split files into (files, operators) runs keyed by their dataset directory.

    Manifest paths are always 'data/<dataset>/...' with forward slashes.
    """
    groups = {}
    for file_path in data_files:
        dataset = file_path.replace(os.sep, '/').split('/')[1]
        groups.setdefault(dataset, []).append(file_path)
    return [(files, operators_by_dataset.get(dataset, fallback))
            for dataset, files in sorted(groups.items())]


def run_suite(args):
    """Run every requested method over a suite. Called by test.py for --suite."""
    from evaluate.instance_timeout import run_instances
    from evaluate.operator_config import set_method_operators
    from evaluate.seed_config import DEFAULT_SEED
    from experiment_runner import check_result_exists, run_experiment, load_method

    ignored = [flag for flag, changed in (
        ('-n/--num-files', args.num_files != -1),
        ('-q/--per-dir-limit', args.per_dir_limit != -1),
        ('-d/--data', bool(args.data)),
    ) if changed]
    if ignored:
        print(" Note: %s ignored - suite '%s' pins its own instance list"
              % (', '.join(ignored), args.suite))

    data_files = load_suite(args.suite)

    print(" Testing %d files from suite '%s'" % (len(data_files), args.suite))
    print(" Files to be processed:")
    print("-" * 60)
    for index, file_path in enumerate(data_files, 1):
        print("%3d. %s" % (index, file_path))
    print("-" * 60)

    if args.list_only:
        return

    # -p/--problem, when given, overrides the operators the suite pins.
    if args.problem:
        run_groups = [(data_files, args.problem)]
    else:
        run_groups = group_by_dataset(data_files,
                                      SUITES[args.suite]['operators'],
                                      args.problem)

    for method_name in args.method:
        print("\n Running method: %s" % method_name)

        if args.timeout > 0:
            for group_files, operators in run_groups:
                for seed in getattr(args, 'seed', None) or [DEFAULT_SEED]:
                    run_instances(method_name, group_files, operators, args, seed,
                                  check_result_exists)
            continue

        method = load_method(method_name)
        if not method:
            continue

        for group_files, operators in run_groups:
            if len(run_groups) > 1:
                print("  group: %d files, operators=%s"
                      % (len(group_files), operators))

            if hasattr(method, 'set_operators'):
                method.set_operators(operators)
            set_method_operators(method_name, operators)

            for seed in getattr(args, 'seed', None) or [DEFAULT_SEED]:
                if len(getattr(args, 'seed', None) or [DEFAULT_SEED]) > 1:
                    print("    seed %d" % seed)

                run_experiment(method,
                               group_files,
                               split=args.split,
                               output_dir=args.output_dir,
                               method_name=method_name,
                               operators=operators,
                               skip_existing=args.skip_existing,
                               use_ground_truth=args.use_ground_truth,
                               seed=seed)
