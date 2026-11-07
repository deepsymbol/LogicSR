"""Generate and verify the Core instance list.

The subset is defined declaratively in logicsr_core/__init__.py and
materialised here as an explicit list of paths, so the exact set is pinned by
a versioned file in the repository rather than by filter arguments that are
easy to mistype.

    python -m logicsr_core.manifest            # write the manifest
    python -m logicsr_core.manifest --check    # verify it matches the spec
"""

import argparse
import collections
import hashlib
import os
import re
import sys

from logicsr_core import (CORE_OPERATORS, CORE_VERSION, DIFFICULTY_ORDER,
                          MANIFEST_NAME, MANIFEST_PATH, MAX_IN, MAX_OUT,
                          REAL_SPEC, REPO_ROOT, SYNTHETIC_SPEC)

IN_OUT_RE = re.compile(r'in(\d+)_out(\d+)$')


def _and_dir_key(name):
    """Sort and10 < and20 < and40 < and80 < and160 numerically, not lexically."""
    match = re.search(r'(\d+)$', name)
    return (int(match.group(1)) if match else -1, name)


def _truth_file_key(name):
    """Sort 1.truth < 2.truth < 10.truth, mirroring test.py's per-dir-limit."""
    stem = name.split('.')[0]
    return (int(stem) if stem.isdigit() else sys.maxsize, name)


def _difficulty_key(name):
    """Order easy < medium < hard, not alphabetically."""
    order = (DIFFICULTY_ORDER.index(name) if name in DIFFICULTY_ORDER
             else len(DIFFICULTY_ORDER))
    return (order, name)


def _in_out_key(name):
    """Sort in5_out5 < in7_out2 < in10_out1 by (inputs, outputs)."""
    match = IN_OUT_RE.match(name)
    return (int(match.group(1)), int(match.group(2))) if match else (1 << 30, 0)


def within_scale_band(name):
    """True when an in<N>_out<M> directory sits inside Core's scale band."""
    match = IN_OUT_RE.match(name)
    return (bool(match) and int(match.group(1)) <= MAX_IN
            and int(match.group(2)) <= MAX_OUT)


def _rel(path):
    return os.path.relpath(path, REPO_ROOT).replace(os.sep, '/')


def collect_synthetic(data_root):
    """Fixed number of lowest-index instances per (scale, gates, noise) cell."""
    files = []
    for suite in SYNTHETIC_SPEC['suites']:
        for in_size in SYNTHETIC_SPEC['in_sizes']:
            in_path = os.path.join(data_root, suite, in_size)
            if not os.path.isdir(in_path):
                raise SystemExit('missing directory: %s' % in_path)

            gate_dirs = sorted(
                (name for name in os.listdir(in_path)
                 if os.path.isdir(os.path.join(in_path, name))),
                key=_and_dir_key)

            for gate_dir in gate_dirs:
                for noise in SYNTHETIC_SPEC['noise_levels']:
                    cell = os.path.join(in_path, gate_dir, noise)
                    if not os.path.isdir(cell):
                        raise SystemExit('missing directory: %s' % cell)

                    truths = sorted(
                        (name for name in os.listdir(cell)
                         if name.endswith('.truth')),
                        key=_truth_file_key)
                    picked = truths[:SYNTHETIC_SPEC['per_dir']]
                    if len(picked) < SYNTHETIC_SPEC['per_dir']:
                        raise SystemExit(
                            '%s holds only %d .truth files, need %d'
                            % (cell, len(picked), SYNTHETIC_SPEC['per_dir']))
                    files.extend(_rel(os.path.join(cell, name))
                                 for name in picked)
    return files


def collect_real(data_root):
    """Every real-circuit instance inside the scale band - no selection rule."""
    files = []
    for suite, noise_levels in sorted(REAL_SPEC['suites'].items()):
        suite_root = os.path.join(data_root, suite)
        if not os.path.isdir(suite_root):
            raise SystemExit('missing directory: %s' % suite_root)

        for noise in (noise_levels if noise_levels else [None]):
            base = suite_root if noise is None else os.path.join(suite_root, noise)
            if not os.path.isdir(base):
                raise SystemExit('missing directory: %s' % base)

            difficulties = sorted(
                (name for name in os.listdir(base)
                 if os.path.isdir(os.path.join(base, name))),
                key=_difficulty_key)

            for difficulty in difficulties:
                dpath = os.path.join(base, difficulty)
                for in_out in sorted(os.listdir(dpath), key=_in_out_key):
                    if not within_scale_band(in_out):
                        continue
                    cell = os.path.join(dpath, in_out)
                    truths = sorted(
                        (name for name in os.listdir(cell)
                         if name.endswith('.truth')),
                        key=_truth_file_key)
                    files.extend(_rel(os.path.join(cell, name))
                                 for name in truths)
    return files


def collect_core_files(data_root):
    """The full Core instance list: synthetic cells plus real circuits."""
    return collect_synthetic(data_root) + collect_real(data_root)


def render_manifest(files):
    """Render the manifest text. Deterministic: no timestamps, no host paths."""
    body = '\n'.join(files) + '\n'
    digest = hashlib.sha256(body.encode('utf-8')).hexdigest()
    per_suite = collections.Counter(path.split('/')[1] for path in files)
    operators = '; '.join('%s -> %s' % (name, ' '.join(ops))
                          for name, ops in sorted(CORE_OPERATORS.items()))
    counts = ', '.join('%s %d' % (name, per_suite[name])
                       for name in sorted(per_suite))

    header = [
        '# LogicSR Core %s - frozen evaluation subset' % CORE_VERSION,
        '#',
        '# Scale band: at most %d inputs and at most %d outputs.' % (MAX_IN, MAX_OUT),
        '#',
        '# Synthetic suites (see logicsr_core/__init__.py):',
        '#   suites        : %s' % ', '.join(SYNTHETIC_SPEC['suites']),
        '#   input sizes   : %s' % ', '.join(SYNTHETIC_SPEC['in_sizes']),
        '#   noise levels  : %s' % ', '.join(SYNTHETIC_SPEC['noise_levels']),
        '#   per directory : first %d files by numeric stem' % SYNTHETIC_SPEC['per_dir'],
        '#',
        '# Real-circuit suites:',
        '#   every instance inside the scale band is taken, so no selection',
        '#   rule applies - the band alone determines the set',
        '#   BBM  : noise levels %s' % ', '.join(REAL_SPEC['suites']['BBM']),
        '#   IWLS : no noise variants',
        '#',
        '# Operators      : %s' % operators,
        '#',
        '# Protocol (see logicsr_core/README.md):',
        '#   train/test split 0.75 with a fixed split seed',
        '#   reported: bit_acc, output_acc, sample_acc (train and test),',
        '#             num_gates, execution_time',
        '#',
        '# instances : %d (%s)' % (len(files), counts),
        '# sha256    : %s' % digest,
        '#',
        '# Run with     : python test.py -m <method> --suite core',
        '# Regenerate   : python -m logicsr_core.manifest',
        '# Verify       : python -m logicsr_core.manifest --check',
        '',
    ]
    return '\n'.join(header) + body


def main():
    parser = argparse.ArgumentParser(
        description='Generate or verify the LogicSR Core instance list.')
    parser.add_argument(
        '--check', action='store_true',
        help='Verify the manifest on disk still matches the spec and the data '
             'tree, without rewriting it. Exits non-zero on mismatch.')
    parser.add_argument(
        '--data-root', default=os.path.join(REPO_ROOT, 'data'),
        help='Dataset root to scan (default: <repo>/data).')
    args = parser.parse_args()

    files = collect_core_files(args.data_root)
    content = render_manifest(files)

    if args.check:
        if not os.path.exists(MANIFEST_PATH):
            raise SystemExit('manifest missing: %s' % MANIFEST_PATH)
        with open(MANIFEST_PATH, encoding='utf-8') as handle:
            on_disk = handle.read()
        if on_disk != content:
            raise SystemExit(
                'manifest does not match the spec or the data tree; '
                'regenerate with: python -m logicsr_core.manifest')
        print('OK: %s matches the spec (%d instances)'
              % (MANIFEST_NAME, len(files)))
        return

    with open(MANIFEST_PATH, 'w', encoding='utf-8') as handle:
        handle.write(content)
    print('wrote logicsr_core/%s (%d instances)' % (MANIFEST_NAME, len(files)))


if __name__ == '__main__':
    main()
