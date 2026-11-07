import numpy as np
from evaluate.data_loader import split_data  
from evaluate.operator_config import get_method_config 
from quine_mccluskey import qm
from evaluate.metrics import calculate_metrics, aggregate_multi_output_metrics


def set_operators(operators):
    config = get_method_config("qm")
    config.set_operators(operators, "QM")


def find_qm_raw_output(X, y, input_size):
    # Extract all minterms where output is 1
    ones = []
    for i in range(len(X)):
        if y[i] == 1:
            index = 0
            for j in range(input_size):
                if X[i][j] == 1:
                    index |= (1 << (input_size - j - 1))
            ones.append(index)

    if not ones:
        return set()

    qm_solver = qm.QuineMcCluskey()
    simplified = qm_solver.simplify(ones=ones, num_bits=input_size)

    # Return raw QM output as set of patterns
    if simplified:
        return set(simplified)
    else:
        return set()


def evaluate_qm_raw_output(qm_output, X):
    if not qm_output:
        return np.zeros(len(X), dtype=int)

    results = np.zeros(len(X), dtype=int)

    for i, x in enumerate(X):
        for pattern in qm_output:
            satisfied = True
            for j, bit in enumerate(pattern):
                if bit != '-':  
                    if j < len(x):
                        if bit == '1':
                            if x[j] != 1:
                                satisfied = False
                                break
                        elif bit == '0':
                            if x[j] != 0:
                                satisfied = False
                                break

            if satisfied:
                results[i] = 1
                break

    return results


def find_expressions(X, Y, split=0.75):
    print("=" * 60)
    print(" QM (Logic Synthesis)")
    print("=" * 60)

    expressions = []
    accuracies = []
    used_vars = set()
    train_pred_columns = []
    test_pred_columns = []

    X_train, X_test, Y_train, Y_test = split_data(X, Y, test_size=1-split)

    for output_idx in range(Y_train.shape[1]):
        y_train = Y_train[:, output_idx]
        y_test = Y_test[:, output_idx]

        print(f" Processing output {output_idx+1}...")

        qm_raw_output = find_qm_raw_output(X_train, y_train,
                                           X_train.shape[1])

        raw_output_str = str(qm_raw_output)

        for pattern in qm_raw_output:
            for i, bit in enumerate(pattern):
                if bit != '-':
                    used_vars.add(f'x{i+1}')

        y_train_pred = evaluate_qm_raw_output(qm_raw_output, X_train)
        y_test_pred = evaluate_qm_raw_output(qm_raw_output, X_test)

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

    extra_info = {
        'all_vars_used': True,
        'aggregated_metrics': aggregated_metrics
    }
    return expressions, accuracies, extra_info
