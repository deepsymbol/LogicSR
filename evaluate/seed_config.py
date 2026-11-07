"""Run-level random seed, injected the way operator_config injects operators.

Methods must not hard-code their search seed: reporting a mean and a spread
over several seeds requires the seed to be settable per run.

The train/test split seed is deliberately NOT part of this. It stays fixed at
SPLIT_SEED so that every method sees identical data - comparisons stay paired -
and so that the noisy and ground-truth splits keep lining up (experiment_runner
relies on both splits using the same random_state). What varies across seeds is
therefore the stochasticity of the search itself, which is what error bars on
this benchmark should express.
"""

import random
import threading

# Seed used when the caller does not ask for anything else. Results produced
# with this seed keep the historical file names, so runs made before seeds were
# configurable are exactly the DEFAULT_SEED runs.
DEFAULT_SEED = 42

# Fixed by protocol - never varied, not even by --seed.
SPLIT_SEED = 42

_lock = threading.Lock()
_run_seed = DEFAULT_SEED


def set_run_seed(seed, method_name="Global"):
    """Set the seed for the run and immediately reseed the global RNGs."""
    global _run_seed
    with _lock:
        _run_seed = int(seed)
    seed_global_rngs(_run_seed)
    print(f"{method_name} seed: {_run_seed}")
    return _run_seed


def get_run_seed():
    """Seed the current run should use for anything stochastic in the search."""
    with _lock:
        return _run_seed


def seed_global_rngs(seed=None):
    """Seed every RNG a method might draw from without taking an explicit seed.

    Most methods do not expose a seed argument and rely on the global numpy or
    torch generators, so seeding those centrally is what makes their runs
    reproducible without editing each method.
    """
    if seed is None:
        seed = get_run_seed()
    seed = int(seed)

    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed % (2 ** 32))
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    return seed


def result_suffix(seed=None):
    """File-name suffix for a seed: empty for DEFAULT_SEED, '__seed<N>' otherwise.

    Keeping DEFAULT_SEED unsuffixed means every result produced before seeds
    were configurable stays valid and is read back as the DEFAULT_SEED run.
    """
    if seed is None:
        seed = get_run_seed()
    return '' if int(seed) == DEFAULT_SEED else '__seed%d' % int(seed)
