"""LogicSR Core - a frozen, versioned subset of the LogicSR benchmark.

Everything that defines, runs and summarises the Core suite lives in this
package; test.py only exposes the --suite flag and hands control over.

    python -m logicsr_core.manifest             # write the instance list
    python -m logicsr_core.manifest --check     # verify it against the spec
    python -m logicsr_core.aggregate            # reference table from output/
    python test.py -m <method> --suite core     # run it

This module holds the declarative spec only, so importing it stays cheap and
free of side effects. See README.md in this directory.
"""

import os

CORE_VERSION = 'v1'

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PACKAGE_DIR)

# Core is a single scale band: at most 10 inputs and at most 10 outputs. The
# synthetic suites hit it by construction (in5_out5, in10_out10); the
# real-circuit suites are filtered down to it.
MAX_IN = 10
MAX_OUT = 10

# Synthetic suites, laid out as data/<suite>/<in_size>/<gates>/<noise>/<n>.truth.
# A fixed number of instances is taken per innermost directory, lowest index
# first, matching what `test.py -q N` selects.
SYNTHETIC_SPEC = {
    'suites': ['AN', 'ANO'],
    'in_sizes': ['in5_out5', 'in10_out10'],
    'noise_levels': ['noise_0', 'noise_0.05'],
    'per_dir': 10,
}

# Real-circuit suites. Every instance inside the scale band is taken, so there
# is no selection rule to argue about - the band alone determines the set.
# BBM is laid out as data/BBM/<noise>/<difficulty>/<in_out>/<n>.truth and has
# noise variants; IWLS is data/IWLS/<difficulty>/<in_out>/<n>.truth and has none.
REAL_SPEC = {
    'suites': {
        'BBM': ['noise_00', 'noise_05'],
        'IWLS': None,
    },
}

DIFFICULTY_ORDER = ['easy', 'medium', 'hard']

# Operator vocabulary each suite is evaluated with. AN instances are built from
# AND/NOT only, so handing a method OR would let it search outside the space
# the dataset was generated from. The real-circuit suites carry no such
# restriction and get the full vocabulary.
CORE_OPERATORS = {
    'AN': ['and', 'not'],
    'ANO': ['and', 'or', 'not'],
    'BBM': ['and', 'or', 'not'],
    'IWLS': ['and', 'not'],
}

MANIFEST_NAME = 'logicsr_core_%s.txt' % CORE_VERSION
MANIFEST_PATH = os.path.join(PACKAGE_DIR, MANIFEST_NAME)

# Suites test.py can be pointed at with --suite.
SUITES = {
    'core': {
        'manifest': MANIFEST_PATH,
        'operators': CORE_OPERATORS,
    },
}
