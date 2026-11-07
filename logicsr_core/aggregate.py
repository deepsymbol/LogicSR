"""Summarise existing per-instance results over the Core subset.

Core is a subset of the full suite, so results that already exist can be
selected by path and summarised without rerunning anything. Coverage is
reported explicitly, so a method that was only partially run shows up as such
instead of being averaged over whatever happens to exist.

    python -m logicsr_core.aggregate
    python -m logicsr_core.aggregate --output-dir output_new --by-cell
"""

import argparse
import collections
import csv
import json
import os
import re
import statistics

from logicsr_core import MANIFEST_PATH, REPO_ROOT
from evaluate.seed_config import DEFAULT_SEED

METRICS = [
    'train_bit_acc', 'test_bit_acc',
    'train_output_acc', 'test_output_acc',
    'train_sample_acc', 'test_sample_acc',
    'num_gates', 'execution_time',
]

# Cells are suite-agnostic: 'group' is the gate count for the synthetic suites
# and the difficulty tier for the real-circuit ones.
CELL_FIELDS = ['suite', 'scale', 'group', 'noise']


def load_manifest(path):
    with open(path, encoding='utf-8') as handle:
        return [line.strip() for line in handle
                if line.strip() and not line.startswith('#')]


def parse_instance(rel_path):
    """Map an instance path to its reporting cell and its result subdirectory.

    Three layouts occur:
        data/ANO/in5_out5/and10/noise_0/1.truth   synthetic
        data/BBM/noise_00/easy/in5_out5/7.truth   real, with noise variants
        data/IWLS/easy/in5_out5/1.truth           real, no noise variants
    """
    parts = rel_path.split('/')
    if len(parts) < 5 or parts[0] != 'data':
        raise ValueError('unexpected instance path: %s' % rel_path)
    suite = parts[1]

    if len(parts) == 6 and parts[2].startswith('in'):
        _, _, scale, group, noise, filename = parts          # AN / ANO
    elif len(parts) == 6:
        _, _, noise, group, scale, filename = parts          # BBM
    elif len(parts) == 5:
        _, _, group, scale, filename = parts                 # IWLS
        noise = '-'
    else:
        raise ValueError('unexpected instance path: %s' % rel_path)

    subdir = os.path.join(*(parts[1:-1] + [os.path.splitext(filename)[0]]))
    return (suite, scale, group, noise), subdir


def split_method_seed(stem):
    """'sr_gplearn__seed7' -> ('sr_gplearn', 7); 'sr_gplearn' -> (name, DEFAULT_SEED)."""
    match = re.match(r'^(?P<method>.+?)__seed(?P<seed>-?\d+)$', stem)
    if match:
        return match.group('method'), int(match.group('seed'))
    return stem, DEFAULT_SEED


def collect(manifest_files, output_dir):
    """Return records[(method, seed, cell)] -> list of per-instance results."""
    records = collections.defaultdict(list)
    skipped = collections.Counter()

    for rel_path in manifest_files:
        cell, subdir = parse_instance(rel_path)
        result_dir = os.path.join(output_dir, subdir)
        if not os.path.isdir(result_dir):
            skipped['<no result directory>'] += 1
            continue
        for name in sorted(os.listdir(result_dir)):
            if not name.endswith('.json'):
                continue
            method, seed = split_method_seed(name[:-len('.json')])
            try:
                with open(os.path.join(result_dir, name), encoding='utf-8') as fh:
                    records[(method, seed, cell)].append(json.load(fh))
            except (ValueError, OSError):
                skipped['%s <unreadable>' % method] += 1
    return records, skipped


def _mean(values):
    numbers = [value for value in values if isinstance(value, (int, float))]
    return (statistics.mean(numbers), len(numbers)) if numbers else (None, 0)


def summarise_metric(per_seed_payloads):
    """Mean over instances, and spread ACROSS seeds.

    per_seed_payloads maps seed -> [metric value per instance]. The reported
    mean pools every instance of every seed; the reported spread is the stdev
    of the per-seed means, i.e. the run-to-run error bar the seeds exist to
    measure. Spread is None with fewer than two seeds that carry the metric.
    """
    seed_means, pooled, total = [], [], 0
    for values in per_seed_payloads.values():
        mean, count = _mean(values)
        if mean is None:
            continue
        seed_means.append(mean)
        pooled.extend(v for v in values if isinstance(v, (int, float)))
        total += count
    if not pooled:
        return None, None, 0, 0
    spread = statistics.stdev(seed_means) if len(seed_means) > 1 else None
    return statistics.mean(pooled), spread, total, len(seed_means)


def build_rows(records, by_cell):
    methods = sorted({method for method, _, _ in records})
    rows = []
    for method in methods:
        # cell -> metric -> {seed: [values]}
        groups = collections.defaultdict(
            lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
        seeds_seen = collections.defaultdict(set)

        for (rec_method, seed, cell), payloads in records.items():
            if rec_method != method:
                continue
            key = cell if by_cell else ('ALL', 'ALL', 'ALL', 'ALL')
            seeds_seen[key].add(seed)
            for metric in METRICS:
                bucket = groups[key][metric][seed]
                bucket.extend(p.get(metric) for p in payloads)

        for cell in sorted(groups):
            row = collections.OrderedDict(zip(CELL_FIELDS, cell))
            row['method'] = method
            row['n_seeds'] = len(seeds_seen[cell])
            # Instance count of one seed (they should match across seeds).
            any_metric = groups[cell]['test_bit_acc']
            row['n_instances'] = max((len(v) for v in any_metric.values()),
                                     default=0)
            for metric in METRICS:
                mean, spread, total, n_seeds = summarise_metric(groups[cell][metric])
                row[metric] = '' if mean is None else round(mean, 6)
                row[metric + '_std'] = '' if spread is None else round(spread, 6)
                row[metric + '_n'] = total
            rows.append(row)
    return rows


def _fmt(value, width=9, places=4):
    if value == '':
        return '--'.rjust(width)
    return ('%%%d.%df' % (width, places)) % value


def _bit_with_spread(row):
    """test_bit mean, with '+-<std>' appended when a seed spread exists."""
    mean = _fmt(row['test_bit_acc'])
    spread = row['test_bit_acc_std']
    return mean if spread == '' else '%s+-%.4f' % (mean.strip(), spread)


def print_summary(rows, by_cell, total_expected):
    if by_cell:
        header = ('%-16s %-5s %-11s %-8s %-9s %5s %4s %-18s %9s %9s %8s %8s'
                  % ('method', 'suite', 'scale', 'group', 'noise', 'n', 'sds',
                     'test_bit(+-seed)', 'test_out', 'test_smpl', 'gates', 'secs'))
    else:
        header = ('%-16s %9s %4s %-18s %9s %9s %8s %8s'
                  % ('method', 'coverage', 'sds', 'test_bit(+-seed)',
                     'test_out', 'test_smpl', 'gates', 'secs'))
    print(header)
    print('-' * len(header))

    for row in rows:
        tail = ('%-18s %s %s %s %s'
                % (_bit_with_spread(row),
                   _fmt(row['test_output_acc']), _fmt(row['test_sample_acc']),
                   _fmt(row['num_gates'], 8, 2),
                   _fmt(row['execution_time'], 8, 2)))
        if by_cell:
            print('%-16s %-5s %-11s %-8s %-9s %5d %4d %s'
                  % (row['method'], row['suite'], row['scale'],
                     row['group'], row['noise'], row['n_instances'],
                     row['n_seeds'], tail))
        else:
            print('%-16s %9s %4d %s'
                  % (row['method'],
                     '%d/%d' % (row['n_instances'], total_expected),
                     row['n_seeds'], tail))


def main():
    parser = argparse.ArgumentParser(
        description='Summarise existing results over the LogicSR Core subset.')
    parser.add_argument(
        '--manifest', default=MANIFEST_PATH,
        help='Core manifest to read (default: the packaged v1 list).')
    parser.add_argument(
        '--output-dir', default='output',
        help='Results directory to aggregate (default: output).')
    parser.add_argument(
        '--by-cell', action='store_true',
        help='Break results down by suite/scale/group/noise instead of one row '
             'per method. Recommended for reporting, since Core mixes noise '
             'levels and both synthetic and real-circuit suites.')
    parser.add_argument(
        '--csv', default=None,
        help='Also write the full table (means, stdevs, per-metric counts) '
             'to this CSV path.')
    args = parser.parse_args()

    manifest_files = load_manifest(args.manifest)
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(REPO_ROOT, output_dir)

    records, skipped = collect(manifest_files, output_dir)
    if not records:
        raise SystemExit('no result JSONs found under %s for Core instances'
                         % args.output_dir)

    rows = build_rows(records, args.by_cell)

    print('Core manifest : %s (%d instances)'
          % (os.path.relpath(args.manifest, REPO_ROOT), len(manifest_files)))
    print('Results from  : %s' % args.output_dir)
    print('Methods found : %d' % len({method for method, _, _ in records}))
    seeds = sorted({seed for _, seed, _ in records})
    print('Seeds found   : %s' % ', '.join(str(s) for s in seeds))
    for reason, count in skipped.most_common():
        print('Skipped       : %s x%d' % (reason, count))
    print()
    print_summary(rows, args.by_cell, len(manifest_files))

    if args.csv:
        csv_path = args.csv
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(REPO_ROOT, csv_path)
        directory = os.path.dirname(csv_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print()
        print('wrote %s (%d rows)'
              % (os.path.relpath(csv_path, REPO_ROOT), len(rows)))


if __name__ == '__main__':
    main()
