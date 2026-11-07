import numpy as np
import re
from typing import Tuple, Dict


def parse_neural_network_complexity(expr: str,
                                    input_size: int) -> Tuple[int, int]:
    """Parse neural network complexity information
    Args:
        expr: Neural network description string
        input_size: Number of input variables
    Returns:
        (num_gates, num_vars) tuple
    """
    if not (expr.startswith("NEURAL_NETWORK_") or expr.startswith("SATNet_")):
        return 0, input_size
    
    # Simple mapping for neural network methods
    if expr == "NEURAL_NETWORK_DiffLogic":
        return input_size, input_size
    elif expr == "NEURAL_NETWORK_NEURALUT":
        return input_size, input_size
    elif expr == "NEURAL_NETWORK_AI4EDA":
        return input_size, input_size
    elif expr.startswith("SATNet_"):
        # SATNet complexity calculation removed - user will calculate separately
        # Default fallback: use input_size
        return input_size, input_size
    else:
        return 0, input_size


def evaluate_expression(expr: str, X: np.ndarray) -> np.ndarray:
    """Evaluate logical expression
    Args:
        expr: Logical expression string
        X: Input data
    Returns:
        Prediction results
    """
    if not expr:
        # Empty expression, return all zeros
        return np.zeros(len(X))

    if expr.startswith('NEURAL_NETWORK_'):
        # For neural network methods, we cannot directly evaluate "expressions"
        # Because they are not true symbolic expressions, but network structure descriptions
        print(
            f"Note: Skip expression evaluation for neural network methods: {expr}"
        )
        return np.zeros(len(X))
    
    if expr.startswith('SATNet_'):
        # For SATNet, we cannot directly evaluate "expressions"
        # Because SATNet learns constraints, not symbolic expressions
        print(
            f"Note: Skip expression evaluation for SATNet: {expr}"
        )
        return np.zeros(len(X))

    # Unified expression evaluation - handle multiple formats
    return evaluate_unified_expression(expr, X)


def evaluate_unified_expression(expr_str, X):
    """Unified expression evaluator that handles multiple formats:
    - Python format: (x1 and x2) or not x3
    - Function format: or(and(x1, x2), not(x3))  
    """
    # For symbolic expressions, continue with original evaluation logic
    results = []
    error_count = 0
    max_errors = 5 

    for row in X:
        # Create variable assignments
        locals_dict = {f'x{i+1}': bool(val) for i, val in enumerate(row)}
        locals_dict.update({
            f'not_x{i+1}': not bool(val)
            for i, val in enumerate(row)
        })

        # Add additional function support
        def not_(x):
            return not bool(x)

        locals_dict.update({
            'not_': not_,
            'True': True,
            'False': False
        })

        try:
            # Replace ~ with not in PyEDA expressions
            cleaned_expr = expr_str.replace('~', ' not ')
            
            # Standard operator replacement
            cleaned_expr = cleaned_expr.replace('and', ' and ').replace('or', ' or ')
            
            # Variable name standardization: replace uppercase X1, X2, etc. with lowercase x1, x2
            cleaned_expr = re.sub(r'X(\d+)', r'x\1', cleaned_expr)
            # Handle not_ function calls
            cleaned_expr = re.sub(r'not_\s+(\w+)', r'not_(\1)', cleaned_expr)

            # If contains And( / Or(, provide corresponding functions
            def And(*args):
                return all(args)

            def Or(*args):
                return any(args)

            locals_dict.update({'And': And, 'Or': Or})

            result = eval(cleaned_expr, {}, locals_dict)
            results.append(1 if result else 0)
        except Exception as e:
            # Only show error messages when error count is low
            if error_count < max_errors:
                print(
                    f"Error evaluating expression: {e}\nExpr: {expr_str}\nCleaned: {cleaned_expr}"
                )
                error_count += 1
            elif error_count == max_errors:
                print(f"... and {len(X) - max_errors} more evaluation errors")
                error_count += 1
            results.append(0)

    return np.array(results)


def output_acc_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Output-bit-level accuracy: each output bit is considered correct only if all samples are correct for that output bit.
    This evaluates expression correctness for each output bit independently (axis=0: along samples).
    Useful for symbolic regression where each output bit has an independent expression.
    """
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    matches = np.all(y_true == y_pred, axis=0)  # axis=0: check all samples for each output bit
    return float(np.mean(matches))


def sample_acc_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Sample-level accuracy: each sample is considered correct only if all output bits are correct.
    This evaluates functional correctness of the entire multi-output function (axis=1: along output bits).
    Useful for evaluating whether the complete function is correct for each input sample.
    """
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    matches = np.all(y_true == y_pred, axis=1)  # axis=1: check all output bits for each sample
    return float(np.mean(matches))


def calculate_metrics(y_true: np.ndarray,
                      y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate evaluation metrics
    Args:
        y_true: True labels
        y_pred: Predicted labels
    Returns:
        Metrics dictionary
    """
    metrics = {
        'bit_acc':
        float(
            np.mean(
                y_pred == y_true
            )),  # Bit-level accuracy (formerly accuracy): compare all bits independently
        'precision':
        float(
            np.sum((y_pred == 1) & (y_true == 1)) /
            (np.sum(y_pred == 1) + 1e-10)),
        'recall':
        float(
            np.sum((y_pred == 1) & (y_true == 1)) /
            (np.sum(y_true == 1) + 1e-10)),
        'sample_acc':
        sample_acc_accuracy(
            y_true, y_pred),  # Sample-level accuracy
        'output_acc':
        output_acc_accuracy(
            y_true, y_pred),  # Output-bit-level accuracy
    }

    metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / (
        metrics['precision'] + metrics['recall'] + 1e-10)

    return metrics


def aggregate_multi_output_metrics(y_train_true: np.ndarray,
                                   y_test_true: np.ndarray,
                                   train_predictions,
                                   test_predictions) -> Dict[str, float]:
    """Aggregate metrics using full multi-output prediction matrices.

    Args:
        y_train_true: Ground-truth training labels (samples x outputs).
        y_test_true: Ground-truth testing labels (samples x outputs).
        train_predictions: Either a numpy array shaped like y_train_true or
            a list of column vectors (one per output).
        test_predictions: Same as train_predictions but for the test split.
    Returns:
        Dictionary containing aggregated metrics for train/test splits, or
        None if predictions are not available.
    """

    def _prepare_matrix(preds):
        if preds is None:
            return None
        if isinstance(preds, np.ndarray):
            if preds.size == 0:
                return None
            return preds if preds.ndim > 1 else preds.reshape(-1, 1)
        preds = list(preds)
        if len(preds) == 0:
            return None
        columns = []
        for col in preds:
            col_array = np.asarray(col)
            if col_array.ndim == 0:
                col_array = np.array([col_array])
            columns.append(col_array.reshape(-1, 1))
        return np.hstack(columns)

    y_train_pred = _prepare_matrix(train_predictions)
    y_test_pred = _prepare_matrix(test_predictions)

    if y_train_pred is None or y_test_pred is None:
        return None

    train_metrics = calculate_metrics(y_train_true, y_train_pred)
    test_metrics = calculate_metrics(y_test_true, y_test_pred)

    return {
        'train_bit_acc': train_metrics['bit_acc'],
        'test_bit_acc': test_metrics['bit_acc'],
        'train_sample_acc': train_metrics['sample_acc'],
        'test_sample_acc': test_metrics['sample_acc'],
        'train_output_acc': train_metrics['output_acc'],
        'test_output_acc': test_metrics['output_acc'],
    }
