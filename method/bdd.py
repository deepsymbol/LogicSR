from dd.autoref import BDD
import numpy as np
from evaluate.data_loader import split_data  
from evaluate.operator_config import get_method_config 
from evaluate.metrics import calculate_metrics, aggregate_multi_output_metrics


def set_operators(operators):
    config = get_method_config("bdd")
    config.set_operators(operators, "BDD")


def create_bdd_from_truth_table(bdd, input_size, X, y):
    """Create BDD expression from truth table"""  
    expr = bdd.false
    for i in range(len(X)):
        if y[i] == 1:
            minterm = bdd.true
            for j in range(input_size):
                var = f'x{j+1}'
                minterm = minterm & (bdd.var(var)
                                     if X[i][j] == 1 else ~bdd.var(var))
            expr = expr | minterm

    return expr


def extract_bdd_raw_output(bdd_expr, bdd):
    """Extract raw BDD output as cube list"""
    if bdd_expr == bdd.false:
        return []
    if bdd_expr == bdd.true:
        return [{}]  

    paths = list(bdd.pick_iter(bdd_expr))
    if not paths:
        return []

    cubes = []
    for path in paths:
        cube = {}
        for var, val in path.items():
            cube[var] = val == 1 
        cubes.append(cube)

    return cubes


def evaluate_bdd_raw_output(bdd_output, X):
    if not bdd_output:
        return np.zeros(len(X), dtype=int)

    results = np.zeros(len(X), dtype=int)

    for i, x in enumerate(X):
        for cube in bdd_output:
            satisfied = True
            for var, val in cube.items():
                var_idx = int(var[1:]) - 1  
                if var_idx < len(x):
                    if val:  
                        if x[var_idx] != 1:
                            satisfied = False
                            break
                    else:  
                        if x[var_idx] != 0:
                            satisfied = False
                            break

            if satisfied:
                results[i] = 1
                break

    return results


def find_expressions(X, Y, split=0.75):
    print("=" * 60)
    print(" BDD (Logic Synthesis)")
    print("=" * 60)

    expressions = []
    accuracies = []
    used_vars = set()
    train_pred_columns = []
    test_pred_columns = []

    X_train, X_test, Y_train, Y_test = split_data(X, Y, test_size=1-split)

    shared_bdd = BDD()
    shared_bdd.configure(reordering=True)
    for v in range(1, X_train.shape[1] + 1):
        shared_bdd.declare(f'x{v}')

    for output_idx in range(Y_train.shape[1]):
        y_train = Y_train[:, output_idx]
        y_test = Y_test[:, output_idx]

        print(f" Processing output {output_idx+1}...")

        expr_node = create_bdd_from_truth_table(shared_bdd,
                                                X_train.shape[1], X_train,
                                                y_train)

        bdd_raw_output = extract_bdd_raw_output(expr_node, shared_bdd)

        raw_output_str = str(bdd_raw_output)

        for cube in bdd_raw_output:
            for var in cube.keys():
                used_vars.add(var)

        y_train_pred = evaluate_bdd_raw_output(bdd_raw_output, X_train)
        y_test_pred = evaluate_bdd_raw_output(bdd_raw_output, X_test)

        train_pred_columns.append(y_train_pred)
        test_pred_columns.append(y_test_pred)

        expressions.append(raw_output_str)

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