import numpy as np
import re
import os
from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import SymbolicRegression
from julia import Main
from evaluate.data_loader import split_data 
from evaluate.metrics import (evaluate_expression, calculate_metrics,
                              aggregate_multi_output_metrics)
from evaluate.operator_config import get_method_config  


if os.environ.get("PYSR_NO_CONSTANT_LEAVES", "1") == "1":
    # On by default; set PYSR_NO_CONSTANT_LEAVES=0 to keep the library's leaf sampling.
    # New leaves are drawn half from constants by default. A constant makes a tree exceed
    # maxsize under complexity_of_constants=1000, and such a tree can hardly mutate any more,
    # so in this constant-free domain leaves are always variables.
    Main.eval("""
    @eval SymbolicRegression.MutationFunctionsModule function make_random_leaf(
        nfeatures::Int, ::Type{T}, ::Type{N},
        rng::AbstractRNG=default_rng(), options::Union{AbstractOptions,Nothing}=nothing,
    ) where {T<:DATA_TYPE,N<:AbstractExpressionNode}
        return constructorof(N)(T; feature=rand(rng, 1:nfeatures))
    end
    """)


def search_settings():
    """Search hyperparameters: (Options keywords, niterations).

    PYSR_PRESET=pysr159 (default) is the main configuration of the paper,
    based on the defaults of PySR 1.5.9; PYSR_PRESET=ours is the reference
    configuration, based on the v0.24.5 defaults. Constants stay disabled in
    both (Boolean domain). PYSR_MAXSIZE, PYSR_POPULATIONS,
    PYSR_POPULATION_SIZE and PYSR_NITERATIONS override either preset;
    PYSR_OPTIONS="key=value,..." sets any other Options keyword.
    """
    preset = os.environ.get("PYSR_PRESET", "pysr159")
    engine_v1 = Main.eval("pkgversion(SymbolicRegression).major >= 1")
    if preset == "ours":
        kwargs = dict(maxsize=20, maxdepth=10, populations=100,
                      ncycles_per_iteration=1000, parsimony=0.1)
        if engine_v1:
            # SymbolicRegression.jl >= 1.0 changed its defaults; keep the 0.24.5 set
            kwargs["defaults"] = Main.eval('v"0.24.5"')
        niterations = 20
    elif preset == "pysr159":
        # SymbolicRegression.jl 1.x defaults equal PySR 1.5.9's, except annealing
        kwargs = dict(annealing=False)
        niterations = 100
    else:
        raise ValueError(f"unknown PYSR_PRESET {preset}")
    kwargs.update(complexity_of_constants=1000, should_optimize_constants=False)

    for env, key in (("PYSR_MAXSIZE", "maxsize"), ("PYSR_POPULATIONS", "populations"),
                     ("PYSR_POPULATION_SIZE", "population_size")):
        if env in os.environ:
            kwargs[key] = int(os.environ[env])
    for item in filter(None, os.environ.get("PYSR_OPTIONS", "").split(",")):
        key, value = item.split("=")
        kwargs[key] = {"true": True, "false": False}.get(value, None)
        if kwargs[key] is None:
            kwargs[key] = float(value) if "." in value or "e" in value else int(value)
    niterations = int(os.environ.get("PYSR_NITERATIONS", niterations))
    return kwargs, niterations


def set_operators(operators):
    config = get_method_config("sr_pysr")
    config.set_operators(operators, "PySR")


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


def evolve_pysr_expression(X_train, y_train, output_idx, full_X_shape):
    # Convert to binary format
    X_train = np.clip(X_train.astype(float).T, 0.0, 1.0)
    y_train = np.clip(y_train.astype(float).flatten(), 0.0, 1.0)

    X_train = (X_train > 0.5).astype(float)
    y_train = (y_train > 0.5).astype(float).flatten()

    config = get_method_config("sr_pysr")

    # Configure binary and unary operators based on available operators
    binary_ops = []
    unary_ops = []
    if config.has_and():
        binary_ops.append(Main.min)
    if config.has_or():
        binary_ops.append(Main.max)
    if config.has_not():
        Main.eval('logical_not(x) = 1.0 - x')
        unary_ops.append(Main.logical_not)

    # Configure PySR search options
    search_kwargs, niterations = search_settings()
    options = SymbolicRegression.Options(
        binary_operators=binary_ops,
        unary_operators=unary_ops,
        verbosity=1,
        progress=False,
        **search_kwargs,
    )

    # Perform symbolic regression search
    equations = SymbolicRegression.equation_search(X_train,
                                                   y_train,
                                                   options=options,
                                                   niterations=niterations)

    # Find the best equation from the evolved population
    best_member = None
    best_loss = float('inf')
    if equations and hasattr(equations, 'members') and len(equations.members) > 0:
        for member in equations.members:
            if hasattr(member, 'loss') and member.loss < best_loss:
                best_member = member
                best_loss = member.loss
    
    # Extract expression string from the best member
    if best_member and hasattr(best_member, 'tree'):
        from julia import Base
        expr_str = Base.string(best_member.tree)
    elif best_member:
        expr_str = str(best_member)
    else:
        expr_str = "False"

    return expr_str


def find_expressions(X, Y, split=0.75, Y_ground_truth=None):
    """Find logic expressions using PySR symbolic regression"""
    print("=" * 60)
    print(" PySR (Symbolic Regression via Julia)")
    print("=" * 60)
    print("SymbolicRegression.jl", Main.eval("string(pkgversion(SymbolicRegression))"),
          "settings:", search_settings())

    expressions = []
    metrics_list = []
    train_pred_columns = []
    test_pred_columns = []
    used_vars = set()

    X_train, X_test, Y_train, Y_test = split_data(X, Y, test_size=1-split)

    for output_idx in range(Y_train.shape[1]):
        y_train = Y_train[:, output_idx]
        y_test = Y_test[:, output_idx]

        # Evolve expression for this output
        expr_str = evolve_pysr_expression(X_train, y_train, output_idx, X.shape)

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

    if Y_ground_truth is not None:
        _, _, Y_train_gt, Y_test_gt = split_data(X, Y_ground_truth, test_size=1-split, random_state=42)
        aggregated_metrics = aggregate_multi_output_metrics(Y_train_gt, Y_test_gt,
                                                            train_pred_columns,
                                                            test_pred_columns)
        print(f"  Using ground truth for evaluation")
    else:
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