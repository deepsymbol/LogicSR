import numpy as np
from evaluate.data_loader import split_data  
from evaluate.metrics import (calculate_metrics, aggregate_multi_output_metrics)  
from evaluate.operator_config import get_method_config  
from pyeda.inter import espresso_tts, exprvar, truthtable


def set_operators(operators):
    config = get_method_config("espresso")
    config.set_operators(operators, "Espresso")


def minimize_multi_with_espresso(X, Y):
    input_size = X.shape[1]
    n_outputs = Y.shape[1]

    # Edge cases: handle constant outputs quickly to avoid espresso call.
    const_mask_zero = np.all(Y == 0, axis=0)
    const_mask_one = np.all(Y == 1, axis=0)

    # Pre-compute row indices of training samples in partial truth table.
    idx_list = []
    for row_bits in X:
        idx = 0
        for b in row_bits[::-1]:
            idx = (idx << 1) | int(b)
        idx_list.append(idx)

    vars = [exprvar(f"x{i+1}") for i in range(input_size)]

    n_rows_full = 1 << input_size

    tt_objects = []
    for col in range(n_outputs):
        if const_mask_zero[col] or const_mask_one[col]:
            tt_objects.append(None)
            continue

        table_chars = ['-'] * n_rows_full  # initialize with don't-care, will be filled with training data
        y_col = Y[:, col]
        for idx_val, out_val in zip(idx_list, y_col):
            table_chars[idx_val] = '1' if out_val == 1 else '0'

        tt_objects.append(truthtable(vars, ''.join(table_chars)))

    active_indices = [
        i for i, tt in enumerate(tt_objects) if tt is not None
    ]
    active_tts = [tt_objects[i] for i in active_indices]

    if active_tts:
        minimized_exprs = list(espresso_tts(*active_tts))
    else:
        minimized_exprs = []

    # Assemble final expression list in original column order
    expressions = []
    active_iter = iter(minimized_exprs)
    for col in range(n_outputs):
        if const_mask_zero[col]:
            expressions.append(exprvar('x1') & ~exprvar('x1'))  
        elif const_mask_one[col]:
            expressions.append(exprvar('x1') | ~exprvar('x1'))  
        else:
            expr_obj = next(active_iter)
            expressions.append(expr_obj)  

    return expressions


def find_expressions(X, Y, split=0.75):
    """Use Espresso method to find logic expressions."""
    print("=" * 60)
    print(" Espresso (Logic Minimization)")
    print("=" * 60)

    expressions = []
    accuracies = []
    used_vars = set()

    X_train, X_test, Y_train, Y_test = split_data(X, Y, test_size=1-split)

    expr_list = minimize_multi_with_espresso(X_train, Y_train)

    train_pred_columns = []
    test_pred_columns = []

    for output_idx, expr in enumerate(expr_list):

        y_test = Y_test[:, output_idx]

        for v in range(1, X.shape[1] + 1):
            if f'x{v}' in str(expr):
                used_vars.add(f'x{v}')

        # Extract corresponding training labels for current output
        y_train = Y_train[:, output_idx]

        y_train_pred = []
        for row in X_train:
            var_dict = {exprvar(f'x{i+1}'): int(val) for i, val in enumerate(row)}
            result = expr.restrict(var_dict)
            y_train_pred.append(1 if bool(result) else 0)
        
        y_test_pred = []
        for row in X_test:
            var_dict = {exprvar(f'x{i+1}'): int(val) for i, val in enumerate(row)}
            result = expr.restrict(var_dict)
            y_test_pred.append(1 if bool(result) else 0)

        y_train_pred = np.array(y_train_pred)
        y_test_pred = np.array(y_test_pred)
        train_pred_columns.append(y_train_pred)
        test_pred_columns.append(y_test_pred)

        expressions.append(str(expr)) 

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
    accuracies = [accuracy_tuple]
    all_vars_used = all(f'x{i}' in used_vars for i in range(1, X.shape[1] + 1))
    extra_info = {
        'all_vars_used': all_vars_used,
        'aggregated_metrics': aggregated_metrics
    }
    return expressions, accuracies, extra_info
