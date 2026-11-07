"""PySR baseline using the native multi-output interface.

This is a variant of ``method/sr_pysr.py``. The only difference is how the
search is invoked: ``sr_pysr.py`` calls ``equation_search`` once per output
with a 1-D target, while this module passes the full (nout, nsamples) target
matrix in a single call, i.e. the multi-output interface of
SymbolicRegression.jl / PySR.

Everything else (operator set, Options, expression conversion, metric
aggregation) is kept byte-for-byte identical so that the two methods form a
controlled A/B comparison for the rebuttal.
"""

import os
import numpy as np
import re
from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import SymbolicRegression
from julia import Main
from evaluate.data_loader import split_data
from evaluate.metrics import (evaluate_expression, calculate_metrics,
                              aggregate_multi_output_metrics)
from evaluate.operator_config import get_method_config
from method.sr_pysr import search_settings

METHOD_NAME = "sr_pysr_multi"


def set_operators(operators):
    config = get_method_config(METHOD_NAME)
    config.set_operators(operators, "PySR (multi-output)")


def convert_pysr_expression(expr_str):
    """Convert PySR expression to logic expression format"""
    if not expr_str or expr_str == "False":
        return "False"

    result = str(expr_str).strip()

    # Convert subscript notation
    def convert_subscript(match):
        subscript = match.group(1)
        # Convert Unicode subscript digits to regular digits
        subscript_map = {
            '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
            '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9'
        }
        converted = ''
        for char in subscript:
            converted += subscript_map.get(char, char)
        return f'x{converted}'

    # Match x followed by any Unicode subscript digits
    result = re.sub(r'x([₀₁₂₃₄₅₆₇₈₉]+)', convert_subscript, result)

    # Convert to evaluate_expression supported format
    result = result.replace("min", "And")
    result = result.replace("max", "Or")
    result = result.replace("logical_not", "not_")

    return result


def build_options():
    """Identical Options to sr_pysr.py, so only the call shape differs."""
    config = get_method_config(METHOD_NAME)

    binary_ops = []
    unary_ops = []
    if config.has_and():
        binary_ops.append(Main.min)
    if config.has_or():
        binary_ops.append(Main.max)
    if config.has_not():
        Main.eval('logical_not(x) = 1.0 - x')
        unary_ops.append(Main.logical_not)

    return SymbolicRegression.Options(
        binary_operators=binary_ops,
        unary_operators=unary_ops,
        verbosity=1,
        progress=False,
        **search_settings()[0],
    )


def _search_kwargs():
    """Optional pinning of the parallelism mode.

    The per-output baseline uses the library default, so this stays unset by
    default. Set LOGICSR_PYSR_PARALLELISM=serial (or multithreading/multi-
    processing) when the wall-clock comparison needs both variants pinned to
    the same scheduling mode.
    """
    kwargs = {}
    mode = os.environ.get('LOGICSR_PYSR_PARALLELISM', '').strip()
    if mode:
        kwargs['parallelism'] = Main.eval(f':{mode}')
    return kwargs


def _best_expression_from_hof(hof):
    """Pick the lowest-loss member of one Hall of Fame (same rule as sr_pysr)."""
    best_member = None
    best_loss = float('inf')

    members = getattr(hof, 'members', None)
    if members is None or len(members) == 0:
        return "False"

    # Newer SymbolicRegression.jl marks unfilled slots via `exists`; when the
    # field is absent the loss of an unfilled slot is Inf and gets skipped.
    exists = getattr(hof, 'exists', None)

    for idx, member in enumerate(members):
        if exists is not None and not exists[idx]:
            continue
        if hasattr(member, 'loss') and member.loss < best_loss:
            best_member = member
            best_loss = member.loss

    if best_member is None:
        return "False"

    if hasattr(best_member, 'tree'):
        from julia import Base
        return Base.string(best_member.tree)
    return str(best_member)


def evolve_pysr_expressions_multi(X_train, Y_train):
    """Run ONE multi-output search and return one expression string per output."""
    n_outputs = Y_train.shape[1]

    # Same binarisation as sr_pysr.py, applied column-wise.
    # SymbolicRegression.jl expects (nfeatures, nsamples) and (nout, nsamples).
    X_jl = np.clip(X_train.astype(float).T, 0.0, 1.0)
    X_jl = (X_jl > 0.5).astype(float)

    Y_jl = np.clip(Y_train.astype(float).T, 0.0, 1.0)
    Y_jl = (Y_jl > 0.5).astype(float)

    options = build_options()

    # The multi-output call: y is a matrix, so the backend allocates one
    # population set and one Hall of Fame per output within a single search.
    hall_of_fames = SymbolicRegression.equation_search(X_jl,
                                                       Y_jl,
                                                       options=options,
                                                       niterations=search_settings()[1],
                                                       **_search_kwargs())

    # Multi-output returns a vector of Hall of Fames (one per output); a
    # single-output degenerate case may return the Hall of Fame directly.
    if hasattr(hall_of_fames, 'members'):
        hall_of_fames = [hall_of_fames]
    else:
        hall_of_fames = list(hall_of_fames)

    if len(hall_of_fames) != n_outputs:
        raise RuntimeError(
            f"multi-output search returned {len(hall_of_fames)} hall of fames "
            f"for {n_outputs} outputs")

    return [_best_expression_from_hof(hof) for hof in hall_of_fames]


def find_expressions(X, Y, split=0.75):
    """Find logic expressions using one multi-output PySR search"""
    print("=" * 60)
    print(" PySR multi-output (Symbolic Regression via Julia)")
    print("=" * 60)

    expressions = []
    metrics_list = []
    train_pred_columns = []
    test_pred_columns = []
    used_vars = set()

    X_train, X_test, Y_train, Y_test = split_data(X, Y, test_size=1-split)

    # One search for all outputs, instead of Y_train.shape[1] separate searches
    raw_expressions = evolve_pysr_expressions_multi(X_train, Y_train)

    for output_idx in range(Y_train.shape[1]):
        expr_str = raw_expressions[output_idx]

        # Convert PySR expression to logic format
        expr = convert_pysr_expression(expr_str)

        y_train_pred = evaluate_expression(expr, X_train)
        y_test_pred = evaluate_expression(expr, X_test)
        train_pred_columns.append(y_train_pred)
        test_pred_columns.append(y_test_pred)

        expressions.append(expr)

        # Extract used variables from expression
        vars_in_expr = re.findall(r'x\d+', expr_str)
        used_vars.update(vars_in_expr)

    aggregated_metrics = aggregate_multi_output_metrics(Y_train, Y_test,
                                                        train_pred_columns,
                                                        test_pred_columns)
    accuracy_tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    if aggregated_metrics:
        accuracy_tuple = (
            aggregated_metrics['train_bit_acc'],
            aggregated_metrics['test_bit_acc'],
            aggregated_metrics['train_sample_acc'],
            aggregated_metrics['test_sample_acc'],
            aggregated_metrics['train_output_acc'],
            aggregated_metrics['test_output_acc'])
    metrics_list = [accuracy_tuple]
    extra_info = {
        'all_vars_used': True,
        'aggregated_metrics': aggregated_metrics
    }
    return expressions, metrics_list, extra_info
