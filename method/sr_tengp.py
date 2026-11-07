import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'external', 'tengp'))

import numpy as np
from typing import List, Tuple
from tengp import FunctionSet, Parameters, simple_es
from evaluate.data_loader import split_data 
from evaluate.metrics import calculate_metrics, aggregate_multi_output_metrics  
from evaluate.operator_config import get_method_config  

import torch
_TORCH_AVAILABLE = torch.cuda.is_available()
_SR_TENGP_USE_GPU_ENV = os.getenv('SR_TENGP_USE_GPU', '').strip().lower()
_SR_TENGP_USE_GPU = _SR_TENGP_USE_GPU_ENV in ('1', 'true', 'yes', 'y')

def set_operators(operators):
    config = get_method_config("sr_tengp")
    config.set_operators(operators, "SR TenGP")


# TenGP Boolean CGP implementation
_DEF_COLS = 80  # CGP grid columns
_DEF_GEN = 600  # evolution iterations
_DEF_LAMBDA = 8  # (1+λ)-ES offspring size


def _make_boolean_funset(use_torch: bool) -> FunctionSet:
    """Generate logical function set based on backend (numpy / torch) and configuration"""
    funset = FunctionSet()

    # Get current operator configuration
    config = get_method_config("sr_tengp")

    if use_torch:
        if config.has_and():
            and_func = lambda a, b: torch.logical_and(a > 0.5, b > 0.5).float()
            and_func.__name__ = 'and'
            funset.add(and_func, 2)
        if config.has_or():
            or_func = lambda a, b: torch.logical_or(a > 0.5, b > 0.5).float()
            or_func.__name__ = 'or'
            funset.add(or_func, 2)
        if config.has_not():
            not_func = lambda a: torch.logical_not(a > 0.5).float()
            not_func.__name__ = 'not'
            funset.add(not_func, 1)
    else:
        if config.has_and():
            and_func = lambda a, b: np.logical_and(a > 0.5, b > 0.5).astype(float)
            and_func.__name__ = 'and'
            funset.add(and_func, 2)
        if config.has_or():
            or_func = lambda a, b: np.logical_or(a > 0.5, b > 0.5).astype(float)
            or_func.__name__ = 'or'
            funset.add(or_func, 2)
        if config.has_not():
            not_func = lambda a: np.logical_not(a > 0.5).astype(float)
            not_func.__name__ = 'not'
            funset.add(not_func, 1)

    return funset


def _convert_expression(expr: str) -> str:
    """Convert TenGP output expression to logicbench unified format"""
    if isinstance(expr, list):
        expr = expr[0] if expr else ''
    if not expr:
        return 'x1'
    
    expr = expr.replace('logical_and', 'and').replace('logical_or', 'or').replace('logical_not', 'not')
    expr = expr.replace('and_', 'and').replace('or_', 'or').replace('not_', 'not')

    import re
    def convert_var(match):
        var_num = int(match.group(1))
        return f"x{var_num + 1}"  # Convert x0->x1, x1->x2, ...
    
    expr = re.sub(r'x(\d+)', convert_var, expr)
    return expr


def _evolve_single_output(
        X_tr, X_te, y_tr, y_te, funset,
        use_torch: bool) -> Tuple[str, np.ndarray, np.ndarray]:

    # Data backend conversion
    if use_torch:
        X_tr_t = torch.tensor(X_tr, device='cuda', dtype=torch.float32)
        X_te_t = torch.tensor(X_te, device='cuda', dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr, device='cuda', dtype=torch.float32)
        y_te_t = torch.tensor(y_te, device='cuda', dtype=torch.float32)
    else:
        X_tr_t, X_te_t = X_tr, X_te
        y_tr_t, y_te_t = y_tr, y_te

    # When using GPU / torch, need to set Parameters.use_tensors to True
    # to avoid transform() internally forcing tensors to NumPy and causing errors
    params = Parameters(
        n_inputs=X_tr.shape[1],
        n_outputs=1,
        n_columns=_DEF_COLS,
        n_rows=1,
        function_set=funset,
        use_tensors=use_torch,
    )

    def _loss(y_true, y_pred):
        if use_torch:
            if isinstance(y_pred, list):
                y_pred = torch.stack(y_pred, dim=1)
            yp_bin = (y_pred > 0.5).float()
            return torch.mean((y_true - yp_bin)**2).item()
        else:
            if isinstance(y_pred, list):
                y_pred = np.stack(y_pred, axis=1)
            yp_bin = (y_pred > 0.5).astype(float)
            return ((y_true - yp_bin)**2).mean()

    # Use (1+λ) strategy — population_size = 1 + λ
    eval_budget = _DEF_GEN * (_DEF_LAMBDA + 1)

    best_population = simple_es(
        X_tr_t,
        y_tr_t,
        cost_function=_loss,
        params=params,
        population_size=_DEF_LAMBDA + 1,
        evaluations=eval_budget,
        random_state=0,
        verbose=False,
    )
    
    # simple_es returns population, last generation list; take best individual
    best = min(best_population, key=lambda ind: ind.fitness)

    # Get expression and convert predictions
    raw_expr = best.get_expression()
    expr = _convert_expression(raw_expr)
    
    train_pred_raw = best.transform(X_tr_t)
    test_pred_raw = best.transform(X_te_t)
    
    # Handle list output from transform
    if isinstance(train_pred_raw, list):
        train_pred_raw = train_pred_raw[0] if train_pred_raw else np.zeros_like(y_tr)
    if isinstance(test_pred_raw, list):
        test_pred_raw = test_pred_raw[0] if test_pred_raw else np.zeros_like(y_te)
    
    # Convert to binary predictions
    if use_torch:
        train_pred = (train_pred_raw > 0.5).int().cpu().numpy()
        test_pred = (test_pred_raw > 0.5).int().cpu().numpy()
    else:
        train_pred = (train_pred_raw > 0.5).astype(int)
        test_pred = (test_pred_raw > 0.5).astype(int)

    return expr, train_pred, test_pred


def find_expressions(X: np.ndarray, Y: np.ndarray, split=0.75, Y_ground_truth=None):
    """Find logic expressions using TenGP
    
    Args:
        X: Input data
        Y: Output data (may be noisy)
        split: Train/test split ratio
        Y_ground_truth: Ground truth output data (noiseless). If provided, 
                       metrics will be calculated using model predictions vs ground truth.
    """
    use_gpu = bool(_TORCH_AVAILABLE and _SR_TENGP_USE_GPU)
    if use_gpu:
        print(" Using GPU acceleration (torch + cuda) [SR_TENGP_USE_GPU enabled]")
    else:
        if _SR_TENGP_USE_GPU and not _TORCH_AVAILABLE:
            print(" CUDA not available, falling back to CPU NumPy backend despite SR_TENGP_USE_GPU being set")
        else:
            print(" Using CPU NumPy backend (default)")

    funset = _make_boolean_funset(use_gpu)

    X_train, X_test, Y_train, Y_test = split_data(X, Y, test_size=1-split, random_state=42)

    expressions: List[str] = []
    metrics_list: List[Tuple[float, float, float, float]] = []
    train_pred_columns: List[np.ndarray] = []
    test_pred_columns: List[np.ndarray] = []
    vars_used = set()

    for col in range(Y.shape[1]):
        expr, train_pred, test_pred = _evolve_single_output(
            X_train,
            X_test,
            Y_train[:, col].astype(float),
            Y_test[:, col].astype(float),
            funset,
            use_gpu,
        )
        
        train_pred_columns.append(train_pred)
        test_pred_columns.append(test_pred)
        expressions.append(expr)
        
        vars_used.update([v for v in range(X.shape[1]) if f"x{v+1}" in expr])

    # If Y_ground_truth is provided, use it for evaluation instead of noisy Y
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
    all_vars_used = (len(vars_used) == X.shape[1])
    extra_info = {
        'all_vars_used': all_vars_used,
        'aggregated_metrics': aggregated_metrics
    }
    return expressions, metrics_list, extra_info
